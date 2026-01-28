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

# Import modul internal
from .globals import CURRENT_CONFIG, ACTIVE_STREAMS
from .plc import update_plc_status
from .notifier import send_whatsapp, send_telegram
import app.core.globals as g

# ==========================================
# KONFIGURASI GLOBAL & MODEL
# ==========================================
# Gunakan 'yolov8n.pt' untuk kecepatan maksimal. 
# Jika PC kuat (ada GPU), ganti ke 'yolov8s.pt' untuk akurasi lebih tinggi.
model = YOLO("yolov8n.pt") 

FOLDER_DATABASE = "database_wajah" 
known_face_encodings = []
known_face_names = []

# --- TUNING PARAMETER (RAHASIA PERFORMA) ---
YOLO_CONFIDENCE = 0.35      # 35% yakin sudah dianggap orang (biar sensitif)
FACE_REC_TOLERANCE = 0.54   # Makin kecil makin ketat. 0.54 adalah sweet spot.
RECHECK_INTERVAL = 15       # Cek wajah ulang setiap 15 frame (0.5 detik) biar gak berat

# ==========================================
# HELPER FUNCTIONS
# ==========================================
def sanitize_image_for_dlib(img_cv):
    if img_cv is None: return None
    # Pastikan format contiguous array (Wajib buat Windows biar gak crash)
    return np.ascontiguousarray(img_cv[:, :, ::-1]) # BGR to RGB shortcut

def load_face_database():
    global known_face_encodings, known_face_names
    
    # Path absolut agar tidak error saat dijalankan dari Flask
    base_dir = os.path.dirname(os.path.abspath(__file__)) # Folder app/
    root_dir = os.path.dirname(base_dir) # Folder project root
    db_path = os.path.join(root_dir, FOLDER_DATABASE)

    if not os.path.exists(db_path):
        os.makedirs(db_path)
        print(f"📂 [INFO] Folder '{db_path}' dibuat.")
        return

    print(f"📂 Memuat Database Wajah dari '{db_path}'...")
    known_face_encodings = [] # Reset dulu
    known_face_names = []
    
    count = 0
    for filename in os.listdir(db_path):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            path = os.path.join(db_path, filename)
            try:
                img = cv2.imread(path)
                rgb = sanitize_image_for_dlib(img)
                
                if rgb is not None:
                    encs = face_recognition.face_encodings(rgb)
                    if encs:
                        name = os.path.splitext(filename)[0].replace("_", " ").upper()
                        name = ''.join([i for i in name if not i.isdigit()]).strip()
                        known_face_encodings.append(encs[0])
                        known_face_names.append(name)
                        count += 1
            except Exception as e:
                print(f"   ❌ Gagal: {filename} ({e})")
                
    print(f"✨ Total Database: {count} wajah.")

# Load saat start
load_face_database()

# ==========================================
# CLASS 1: BUFFERLESS VIDEO CAPTURE (ANTI LAG)
# ==========================================
class BufferlessVideoCapture:
    def __init__(self, name):
        self.cap = cv2.VideoCapture(name)
        # Setting Buffer Size kecil untuk RTSP
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        if isinstance(name, int):
            self.cap.set(3, 640) # Width
            self.cap.set(4, 480) # Height
            
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
                time.sleep(0.5) # Tunggu sebentar jika putus
                continue
            
            # KUNCI ANTI LAG: Selalu timpa frame lama dengan yang baru
            with self.lock:
                self.latest_frame = frame
                self.status = True

    def read(self):
        with self.lock:
            if self.latest_frame is not None:
                return self.status, self.latest_frame.copy()
            return self.status, None

    def release(self):
        self.running = False
        self.t.join()
        self.cap.release()

# ==========================================
# CLASS 2: HYBRID CAM STREAM
# ==========================================
class CamStream(threading.Thread):
    def __init__(self, cam_id, source):
        threading.Thread.__init__(self)
        self.cam_id = cam_id
        self.source = int(source) if str(source).isdigit() else source
        self.running = True
        self.output_frame = None
        
        self.detected_ids = set() 
        self.last_detection_time = 0
        self.local_lock = threading.Lock()
        self.cap = None 

        # Cache Identitas: { track_id: "NAMA" }
        self.identity_cache = {}
        self.frame_count = 0

    def run(self):
        print(f"🎥 Start Cam {self.cam_id}...")
        self.cap = BufferlessVideoCapture(self.source)
        time.sleep(1) 

        while self.running:
            success, frame = self.cap.read()
            
            # Reconnect Logic
            if not success or frame is None:
                print(f"⚠️ Cam {self.cam_id} Reconnecting...")
                time.sleep(2)
                self.cap.release()
                self.cap = BufferlessVideoCapture(self.source)
                time.sleep(1)
                continue
            
            try:
                # 1. YOLO TRACKING (Engine Utama)
                # conf=YOLO_CONFIDENCE (0.35) agar lebih peka
                results = model.track(frame, persist=True, classes=[0], conf=YOLO_CONFIDENCE, verbose=False)
                
                boxes = []
                track_ids = []
                
                if results[0].boxes.id is not None:
                    boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
                    track_ids = results[0].boxes.id.cpu().numpy().astype(int)
                
                has_person = len(boxes) > 0
                update_plc_status(self.cam_id, has_person)

                annotated_frame = frame.copy()

                if has_person:
                    for box, track_id in zip(boxes, track_ids):
                        x1, y1, x2, y2 = box
                        
                        # --- LOGIKA PENGENALAN WAJAH (OPTIMAL) ---
                        # Cek wajah HANYA jika:
                        # 1. ID ini belum dikenal (cache kosong)
                        # 2. ATAU Sudah waktunya re-check (setiap 15 frame)
                        should_recognize = (track_id not in self.identity_cache) or (self.frame_count % RECHECK_INTERVAL == 0)
                        
                        if should_recognize:
                            # [PERBAIKAN] Jangan crop 40% atas saja.
                            # Ambil kotak badan dengan padding, biarkan Face Recog mencari wajahnya.
                            h, w = frame.shape[:2]
                            pad_x = int((x2 - x1) * 0.1) # Padding 10%
                            pad_y = int((y2 - y1) * 0.1)
                            
                            crop_x1 = max(0, x1 - pad_x)
                            crop_y1 = max(0, y1 - pad_y)
                            crop_x2 = min(w, x2 + pad_x)
                            # Ambil sampai 60% badan ke bawah (cukup sampai dada)
                            crop_y2 = min(h, y1 + int((y2 - y1) * 0.6)) 
                            
                            face_crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]
                            
                            # Validasi crop tidak kosong
                            if face_crop.size > 0:
                                rgb_crop = sanitize_image_for_dlib(face_crop)
                                
                                # Detect Wajah dalam Crop
                                # model="hog" lebih cepat dari "cnn"
                                face_locs = face_recognition.face_locations(rgb_crop, model="hog")
                                
                                name = "UNKNOWN"
                                if face_locs:
                                    # Ambil encoding wajah pertama (terbesar)
                                    face_enc = face_recognition.face_encodings(rgb_crop, face_locs)[0]
                                    
                                    if len(known_face_encodings) > 0:
                                        matches = face_recognition.compare_faces(known_face_encodings, face_enc, tolerance=FACE_REC_TOLERANCE)
                                        dists = face_recognition.face_distance(known_face_encodings, face_enc)
                                        
                                        if True in matches:
                                            best_match_idx = np.argmin(dists)
                                            if matches[best_match_idx]:
                                                name = known_face_names[best_match_idx]
                                
                                # Simpan hasil ke Cache
                                self.identity_cache[track_id] = name
                            else:
                                # Jika crop gagal, jangan update cache (biar dicoba lagi frame depan)
                                pass

                        # --- VISUALISASI ---
                        display_name = self.identity_cache.get(track_id, "Scanning...")
                        
                        # Warna: Hijau (Kenal), Merah (Unknown/Scanning)
                        color = (0, 255, 0) if display_name not in ["UNKNOWN", "Scanning..."] else (0, 0, 255)
                        
                        # Gambar Kotak
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                        
                        # Label di atas kotak
                        label = f"{display_name} [{track_id}]"
                        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                        cv2.rectangle(annotated_frame, (x1, y1 - 25), (x1 + t_size[0] + 10, y1), color, -1)
                        cv2.putText(annotated_frame, label, (x1 + 5, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                    # Kirim Notifikasi
                    self.handle_alert(track_ids, annotated_frame)
                
                # Bersihkan cache ID lama
                active_ids = set(track_ids)
                expired = [k for k in self.identity_cache if k not in active_ids]
                for k in expired: del self.identity_cache[k]

                # Update Output Frame
                with self.local_lock:
                    self.output_frame = annotated_frame.copy()
                
                self.frame_count += 1
            
            except Exception as e:
                # print(f"❌ Logic Error: {e}") # Uncomment untuk debug
                with self.local_lock:
                    self.output_frame = frame

        if self.cap: self.cap.release()
        print(f"🛑 Cam {self.cam_id} Stopped")

    def handle_alert(self, current_ids, frame):
        now = time.time()
        cooldown = int(CURRENT_CONFIG.get('cooldown', 30))
        
        # Reset memori jika sepi > 30 detik
        if (now - self.last_detection_time > cooldown):
            self.detected_ids.clear()
        
        self.last_detection_time = now
        if len(current_ids) == 0: return
        
        # Cek Global Cooldown Notifikasi
        if (now - g.last_global_send_time > cooldown):
            new_person = False
            for pid in current_ids:
                if pid not in self.detected_ids:
                    self.detected_ids.add(pid)
                    new_person = True
            
            if new_person:
                g.last_global_send_time = now
                print(f"🔔 Notif Sent from Cam {self.cam_id}")
                
                # Compress gambar (Quality 60%) biar kirimnya cepat
                _, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 60])
                
                if CURRENT_CONFIG.get('waha_enabled'):
                    b64 = base64.b64encode(buffer).decode('utf-8')
                    threading.Thread(target=send_whatsapp, args=(self.cam_id, len(current_ids), b64)).start()
                
                if CURRENT_CONFIG.get('telegram_enabled'):
                    img_bytes = buffer.tobytes()
                    threading.Thread(target=send_telegram, args=(self.cam_id, len(current_ids), img_bytes)).start()

    def get_frame(self):
        with self.local_lock:
            if self.output_frame is None: return None
            
            # RESIZE DISPLAY (Biar Web Interface Ringan)
            h, w = self.output_frame.shape[:2]
            if w > 640:
                scale = 640 / float(w)
                out = cv2.resize(self.output_frame, None, fx=scale, fy=scale)
            else:
                out = self.output_frame
            
            ret, encoded = cv2.imencode(".jpg", out, [int(cv2.IMWRITE_JPEG_QUALITY), 50])
            return bytearray(encoded) if ret else None
    
    def stop(self):
        self.running = False
        self.join()

def restart_camera_threads():
    for stream in ACTIVE_STREAMS.values(): stream.stop()
    ACTIVE_STREAMS.clear()
    cams = CURRENT_CONFIG.get('cameras', {})
    for cid, src in cams.items():
        if src and str(src).strip():
            stream = CamStream(cid, src)
            stream.daemon = True
            stream.start()
            ACTIVE_STREAMS[cid] = stream
