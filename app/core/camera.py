import os
# [PENTING] Paksa FFmpeg menggunakan TCP untuk RTSP agar gambar tidak rusak/abu-abu
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

import cv2
import time
import threading
import base64
import queue
import numpy as np
import face_recognition
from ultralytics import YOLO

# Import modul internal (Sesuai struktur project Anda)
from .globals import CURRENT_CONFIG, ACTIVE_STREAMS
from .plc import update_plc_status
from .notifier import send_whatsapp, send_telegram
import app.core.globals as g

# ==========================================
# KONFIGURASI GLOBAL & MODEL
# ==========================================
# Load model YOLO sekali saja di global scope
model = YOLO("yolov8n.pt")

# Konfigurasi Database Wajah
FOLDER_DATABASE = "database_wajah" 
known_face_encodings = []
known_face_names = []

# ==========================================
# HELPER FUNCTIONS
# ==========================================
def sanitize_image_for_dlib(img_cv):
    """
    Membersihkan format gambar agar bisa dibaca library dlib/face_recognition di Windows.
    Mengubah ke RGB dan memastikan memori contiguous.
    """
    if img_cv is None: return None
    img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
    img_clean = np.ascontiguousarray(img_rgb, dtype=np.uint8)
    return img_clean

def load_face_database():
    """
    Membaca semua foto di folder database_wajah saat program dimulai.
    """
    global known_face_encodings, known_face_names
    
    if not os.path.exists(FOLDER_DATABASE):
        os.makedirs(FOLDER_DATABASE)
        print(f"📂 Membuat folder '{FOLDER_DATABASE}'. Silakan isi foto wajah.")
        return

    print(f"📂 Memuat Database Wajah dari '{FOLDER_DATABASE}'...")
    loaded_count = 0
    
    for filename in os.listdir(FOLDER_DATABASE):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            path = os.path.join(FOLDER_DATABASE, filename)
            try:
                img = cv2.imread(path)
                clean_img = sanitize_image_for_dlib(img)
                
                if clean_img is not None:
                    encs = face_recognition.face_encodings(clean_img)
                    if encs:
                        # Ambil nama file sebagai nama orang (hapus extension dan angka)
                        name = os.path.splitext(filename)[0].replace("_", " ").upper()
                        name = ''.join([i for i in name if not i.isdigit()]).strip()
                        
                        known_face_encodings.append(encs[0])
                        known_face_names.append(name)
                        loaded_count += 1
                        print(f"   ✅ Terdaftar: {name}")
            except Exception as e:
                print(f"   ❌ Gagal load {filename}: {e}")
                
    print(f"✨ Total Database: {loaded_count} wajah siap dikenali.")

# Panggil fungsi load database saat modul ini di-import
load_face_database()

# ==========================================
# CLASS 1: BUFFERLESS VIDEO CAPTURE
# ==========================================
class BufferlessVideoCapture:
    def __init__(self, name):
        self.cap = cv2.VideoCapture(name)
        
        # Optimize Webcam (Jika USB Cam)
        if isinstance(name, int):
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            
        self.lock = threading.Lock()
        self.t = threading.Thread(target=self._reader)
        self.t.daemon = True
        self.running = True
        self.latest_frame = None
        self.status = False
        self.t.start()

    def _reader(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                with self.lock:
                    self.status = False
                time.sleep(0.5)
                continue
            
            # Simpan frame terbaru saja (Drop frame lama)
            with self.lock:
                self.latest_frame = frame
                self.status = True

    def read(self):
        with self.lock:
            # Kembalikan copy frame agar thread safe
            if self.latest_frame is not None:
                return self.status, self.latest_frame.copy()
            return self.status, None

    def release(self):
        self.running = False
        self.t.join()
        self.cap.release()

# ==========================================
# CLASS 2: CAMERA STREAM PROCESSOR (HYBRID)
# ==========================================
class CamStream(threading.Thread):
    def __init__(self, cam_id, source):
        threading.Thread.__init__(self)
        self.cam_id = cam_id
        self.source = int(source) if str(source).isdigit() else source
        self.running = True
        self.output_frame = None
        
        # Set untuk notifikasi (Global cooldown logic)
        self.detected_ids = set() 
        self.last_detection_time = 0
        
        self.local_lock = threading.Lock()
        self.cap = None 

        # --- VARIABEL LOGIKA HYBRID ---
        # Cache Identitas: { track_id_yolo: "NAMA_ORANG" }
        self.identity_cache = {}
        self.frame_count = 0

    def run(self):
        print(f"🎥 Start Cam {self.cam_id} on {self.source}")
        
        self.cap = BufferlessVideoCapture(self.source)
        time.sleep(1) # Warmup

        while self.running:
            success, frame = self.cap.read()
            
            # 1. Validasi Frame
            if not success or frame is None or frame.size == 0:
                print(f"⚠️ Cam {self.cam_id} Reconnecting...")
                time.sleep(2)
                self.cap.release()
                self.cap = BufferlessVideoCapture(self.source)
                time.sleep(1)
                continue
            
            try:
                # Ambil confidence dari config
                conf = float(CURRENT_CONFIG.get('confidence', 0.5))
                
                # 2. YOLO TRACKING (Engine Utama)
                # persist=True agar ID (Track ID) konsisten mengikuti orangnya
                results = model.track(frame, persist=True, classes=[0], conf=conf, verbose=False)
                
                # Ambil data deteksi
                boxes = []
                track_ids = []
                
                if results[0].boxes.id is not None:
                    boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
                    track_ids = results[0].boxes.id.cpu().numpy().astype(int)
                
                has_person = len(boxes) > 0

                # 3. Update PLC Status
                update_plc_status(self.cam_id, has_person)

                # 4. LOGIKA FACE RECOGNITION (Hanya pada crop wajah)
                # Kita akan menggambar anotasi manual (custom drawing)
                annotated_frame = frame.copy()

                if has_person:
                    # Loop setiap orang yang terdeteksi YOLO
                    for box, track_id in zip(boxes, track_ids):
                        x1, y1, x2, y2 = box
                        
                        # Cek Identitas (Logic Optimization):
                        # Jalankan Face Recog HANYA JIKA:
                        # a. ID ini belum ada di cache (Orang baru masuk)
                        # b. Sudah 30 frame berlalu sejak cek terakhir (Re-verifikasi berkala)
                        if (track_id not in self.identity_cache) or (self.frame_count % 30 == 0):
                            
                            # -- SMART CROP --
                            # Ambil 40% bagian atas kotak badan (Area Kepala & Dada)
                            h_box = y2 - y1
                            crop_y2 = y1 + int(h_box * 0.40)
                            
                            # Clamp agar tidak keluar gambar
                            crop_y2 = min(crop_y2, frame.shape[0])
                            crop_y1 = max(0, y1)
                            crop_x1 = max(0, x1)
                            crop_x2 = min(frame.shape[1], x2)
                            
                            face_crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]
                            
                            # Jalankan Face Recog pada potongan kecil ini
                            rgb_crop = sanitize_image_for_dlib(face_crop)
                            
                            name = "UNKNOWN"
                            if rgb_crop is not None and rgb_crop.size > 0:
                                # Cari lokasi wajah di dalam crop
                                face_locs = face_recognition.face_locations(rgb_crop)
                                
                                if face_locs:
                                    # Encode wajah
                                    face_enc = face_recognition.face_encodings(rgb_crop, face_locs)[0]
                                    
                                    # Bandingkan dengan database
                                    if len(known_face_encodings) > 0:
                                        matches = face_recognition.compare_faces(known_face_encodings, face_enc, tolerance=0.50)
                                        dists = face_recognition.face_distance(known_face_encodings, face_enc)
                                        
                                        if True in matches:
                                            best_match_idx = np.argmin(dists)
                                            name = known_face_names[best_match_idx]
                            
                            # Simpan hasil ke Cache
                            self.identity_cache[track_id] = name
                        
                        # -- VISUALISASI --
                        # Ambil nama dari cache
                        display_name = self.identity_cache.get(track_id, "Scanning...")
                        
                        # Warna: Hijau jika kenal, Merah jika Unknown
                        color = (0, 255, 0) if display_name != "UNKNOWN" and display_name != "Scanning..." else (0, 0, 255)
                        
                        # Gambar Kotak Badan
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                        
                        # Gambar Label Nama
                        label = f"{display_name} ({track_id})"
                        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                        cv2.rectangle(annotated_frame, (x1, y1 - 25), (x1 + w, y1), color, -1)
                        cv2.putText(annotated_frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                    # Handle Notifikasi (Kirim WA/Telegram dengan frame yang sudah ada namanya)
                    self.handle_alert(track_ids, annotated_frame)
                
                else:
                    # Jika tidak ada orang, kosongkan frame anotasi
                    annotated_frame = frame

                # Bersihkan cache ID yang sudah hilang (Memory Cleanup)
                active_ids = set(track_ids)
                expired_ids = [k for k in self.identity_cache if k not in active_ids]
                for k in expired_ids:
                    del self.identity_cache[k]

                # 5. Update Frame untuk Web View
                with self.local_lock:
                    self.output_frame = annotated_frame.copy()
                
                self.frame_count += 1
            
            except Exception as e:
                print(f"❌ Error Logic Cam {self.cam_id}: {e}")
                # Fallback jika error, tampilkan frame polos
                with self.local_lock:
                    self.output_frame = frame

        if self.cap:
            self.cap.release()
        print(f"🛑 Cam {self.cam_id} Stopped")

    def handle_alert(self, current_ids, frame):
        """
        Menangani pengiriman Notifikasi (WA/Telegram).
        Sekarang menerima `current_ids` (List ID) dan `frame` (Gambar).
        """
        now = time.time()
        cooldown = int(CURRENT_CONFIG.get('cooldown', 30))
        
        # Reset memori jika sudah lama kosong (misal 30 detik tidak ada orang)
        if (now - self.last_detection_time > cooldown):
            self.detected_ids.clear()
        
        # Update waktu terakhir deteksi
        self.last_detection_time = now

        if len(current_ids) == 0: return
        
        new_detection = False
        
        # Cek Global Cooldown (Agar WA tidak spam)
        if (now - g.last_global_send_time > cooldown):
            for pid in current_ids:
                if pid not in self.detected_ids:
                    self.detected_ids.add(pid)
                    new_detection = True
            
            if new_detection:
                g.last_global_send_time = now
                print(f"🔔 Notif Triggered by Cam {self.cam_id}")
                
                count = len(current_ids)
                
                # Compress gambar untuk pengiriman
                _, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 60])
                
                # Kirim WhatsApp
                if CURRENT_CONFIG.get('waha_enabled'):
                    b64 = base64.b64encode(buffer).decode('utf-8')
                    threading.Thread(target=send_whatsapp, args=(self.cam_id, count, b64)).start()
                
                # Kirim Telegram
                if CURRENT_CONFIG.get('telegram_enabled'):
                    img_bytes = buffer.tobytes()
                    threading.Thread(target=send_telegram, args=(self.cam_id, count, img_bytes)).start()

    def get_frame(self):
        with self.local_lock:
            if self.output_frame is None: return None
            
            # Optimasi Resize untuk Web Stream (Max width 640px)
            height, width = self.output_frame.shape[:2]
            target_width = 640
            
            if width > target_width:
                scale = target_width / float(width)
                display_frame = cv2.resize(self.output_frame, None, fx=scale, fy=scale)
            else:
                display_frame = self.output_frame

            # Compress JPEG
            ret, encoded = cv2.imencode(".jpg", display_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 50])
            return bytearray(encoded) if ret else None
    
    def stop(self):
        self.running = False
        self.join()

def restart_camera_threads():
    for stream in ACTIVE_STREAMS.values():
        stream.stop()
    ACTIVE_STREAMS.clear()
    
    cams = CURRENT_CONFIG.get('cameras', {})
    for cid, src in cams.items():
        if src and str(src).strip():
            stream = CamStream(cid, src)
            stream.daemon = True
            stream.start()
            ACTIVE_STREAMS[cid] = stream
