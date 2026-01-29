# Lokasi File: app/core/camera.py

import os
# [PENTING] Paksa FFmpeg menggunakan TCP untuk RTSP agar gambar tidak rusak/abu-abu
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

import cv2
import time
import threading
import base64
import torch
import numpy as np
from ultralytics import YOLO

# --- IMPORT MODUL INTERNAL ---
from .globals import CURRENT_CONFIG, ACTIVE_STREAMS
from .plc import update_plc_status
from .notifier import send_whatsapp, send_telegram
from .face_engine import FaceEngine  # Modul Face Recognition Hybrid
import app.core.globals as g

# ==========================================
# 1. SETUP ENGINE AI (HYBRID MODE)
# ==========================================

# A. Setup YOLO (GPU - GTX 745)
# Kita gunakan GPU untuk deteksi objek karena butuh FPS tinggi
if torch.cuda.is_available():
    DEVICE = 'cuda:0'
    print(f"🚀 YOLO Engine: GPU {torch.cuda.get_device_name(0)}")
    print(f"ℹ️  Capability: {torch.cuda.get_device_capability(0)} | CUDA: {torch.version.cuda}")
else:
    DEVICE = 'cpu'
    print("⚠️ YOLO Engine: CPU (Fallback - Performa Terbatas)")

# Load Model YOLO
try:
    # Menggunakan model Nano (n) agar VRAM 2GB/4GB cukup
    model = YOLO("yolov8n.pt")
    model.to(DEVICE)
except Exception as e:
    print(f"❌ Error Load YOLO: {e}")
    model = YOLO("yolov8n.pt")

# B. Setup Face Engine (CPU - i7 Gen 4)
# Folder 'known_faces' harus ada di root project berisi foto wajah (arya.jpg, dll)
print("🧠 Menyiapkan Face Engine...")
face_engine = FaceEngine("known_faces") 


# ==========================================
# 2. CLASS BUFFERLESS CAPTURE
# ==========================================
class BufferlessVideoCapture:
    """
    Kelas ini memastikan kita selalu mengambil frame TERBARU dari kamera,
    membuang frame lama yang menumpuk di buffer (anti-lag).
    """
    def __init__(self, name):
        self.cap = cv2.VideoCapture(name)
        
        # Optimize Webcam (Jika inputnya index USB Cam)
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
                time.sleep(0.5) # Tunggu sebentar jika sinyal putus
                continue
            
            # Selalu simpan frame terakhir saja
            with self.lock:
                self.latest_frame = frame
                self.status = True

    def read(self):
        with self.lock:
            return self.status, self.latest_frame

    def release(self):
        self.running = False
        self.t.join()
        self.cap.release()

# ==========================================
# 3. CLASS CAM STREAM (PROCESSOR UTAMA)
# ==========================================
class CamStream(threading.Thread):
    def __init__(self, cam_id, source):
        threading.Thread.__init__(self)
        self.cam_id = cam_id
        self.source = int(source) if str(source).isdigit() else source
        self.running = True
        self.output_frame = None
        
        # --- MEMORI NOTIFIKASI ---
        self.detected_ids = set() 
        self.last_detection_time = 0
        
        # --- MEMORI WAJAH ---
        self.face_cache = {}    # Format: { id_yolo : "Nama" }
        self.frame_count = 0    
        self.rec_interval = 10  # Cek wajah setiap 10 frame (Hybrid Optimization)
        
        self.local_lock = threading.Lock()
        self.cap = None 

    def run(self):
        print(f"🎥 Start Cam {self.cam_id} [Hybrid Mode Active]")
        self.cap = BufferlessVideoCapture(self.source)
        time.sleep(1) # Warmup camera

        reset_tracker = True 
        
        while self.running:
            success, frame = self.cap.read()
            
            # 1. Validasi & Reconnect
            if not success or frame is None or frame.size == 0:
                print(f"⚠️ Cam {self.cam_id} Lost Signal... Reconnecting")
                time.sleep(2)
                self.cap.release()
                self.cap = BufferlessVideoCapture(self.source)
                reset_tracker = True 
                continue
            
            self.frame_count += 1
            
            try:
                annotated_frame = frame.copy()
                has_person = False
                conf = float(CURRENT_CONFIG.get('confidence', 0.5))

                # 2. YOLO TRACKING (GPU)
                # imgsz=640: Kualitas HD agar wajah terlihat jelas
                # half=False: Wajib False untuk kestabilan GTX 745 (Maxwell)
                results = model.track(
                    source=frame,
                    persist=not reset_tracker,
                    classes=[0],       # Hanya Person
                    conf=conf,
                    imgsz=640,         # Resolusi HD
                    verbose=False,
                    device=DEVICE,
                    half=False,        
                    stream=False
                )
                
                if reset_tracker: reset_tracker = False

                # 3. LOGIKA DETEKSI & WAJAH
                if results[0].boxes and results[0].boxes.id is not None:
                    has_person = True
                    
                    boxes = results[0].boxes.xyxy.cpu().numpy()
                    track_ids = results[0].boxes.id.cpu().numpy().astype(int)

                    for box, track_id in zip(boxes, track_ids):
                        x1, y1, x2, y2 = map(int, box)
                        
                        # --- HYBRID FACE RECOGNITION (CPU) ---
                        # Ambil nama dari cache dulu (cepat)
                        detected_name = self.face_cache.get(track_id, "Unknown")
                        
                        # Kapan harus scan wajah ulang?
                        # a. Waktunya interval (frame ke-10, 20, ...)
                        # b. ATAU ID ini benar-benar baru
                        is_time_check = (self.frame_count % self.rec_interval == 0)
                        is_new_id = (track_id not in self.face_cache)
                        
                        # Filter Jarak: Hanya scan jika kotak tinggi > 100px (Orang cukup dekat)
                        box_h = y2 - y1

                        if (is_time_check or is_new_id) and box_h > 100:
                            
                            # Crop area badan/wajah dengan aman (cegah koordinat minus)
                            h_img, w_img = frame.shape[:2]
                            c_x1, c_y1 = max(0, x1), max(0, y1)
                            c_x2, c_y2 = min(w_img, x2), min(h_img, y2)
                            
                            body_crop = frame[c_y1:c_y2, c_x1:c_x2]
                            
                            if body_crop.size > 0:
                                # Lempar ke CPU (InsightFace)
                                name_result = face_engine.recognize_crop(body_crop)
                                
                                # Update Cache
                                self.face_cache[track_id] = name_result
                                detected_name = name_result
                        
                        # --- VISUALISASI ---
                        # Warna: Hijau (Dikenal), Merah (Unknown)
                        color = (0, 255, 0) if detected_name != "Unknown" else (0, 0, 255)
                        
                        # Gambar Kotak
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                        
                        # Gambar Label (ID + Nama) dengan background hitam
                        label = f"#{track_id} {detected_name}"
                        (w_text, h_text), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                        cv2.rectangle(annotated_frame, (x1, y1 - 25), (x1 + w_text, y1), color, -1)
                        cv2.putText(annotated_frame, label, (x1, y1 - 5), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                    # 4. HANDLE NOTIFIKASI
                    # Kita kirim hasil tracking untuk diproses logikanya
                    self.handle_alert(results[0], annotated_frame)

                # 5. UPDATE PLC (Real-time)
                update_plc_status(self.cam_id, has_person)
                
                # 6. UPDATE WEB VIEW
                with self.local_lock:
                    self.output_frame = annotated_frame 
            
            except Exception as e:
                # Handle error GPU OOM (Out Of Memory)
                if "out of memory" in str(e).lower():
                    torch.cuda.empty_cache()
                    print(f"⚠️ GPU OOM Cam {self.cam_id} - Clearing Cache")
                else:
                    print(f"❌ Error Runtime Cam {self.cam_id}: {e}")
                
                reset_tracker = True
                with self.local_lock:
                    self.output_frame = frame

        if self.cap:
            self.cap.release()
        print(f"🛑 Cam {self.cam_id} Stopped")

    def handle_alert(self, result, frame):
        """
        Logika Pengiriman Notifikasi Cerdas dengan Nama.
        """
        now = time.time()
        cooldown = int(CURRENT_CONFIG.get('cooldown', 30))
        
        # Reset memori lokal jika scene sepi lama
        if (now - self.last_detection_time > cooldown):
            self.detected_ids.clear()
        
        self.last_detection_time = now

        if result.boxes.id is None: return
        
        current_ids = result.boxes.id.cpu().numpy().astype(int)
        
        # Cek Global Cooldown (Anti Spam WA)
        if (now - g.last_global_send_time < cooldown):
            return

        # Cek apakah ada ID BARU?
        new_detection = False
        detected_names = []

        for pid in current_ids:
            # Ambil nama dari cache
            name = self.face_cache.get(pid, "Unknown")
            detected_names.append(name)

            if pid not in self.detected_ids:
                self.detected_ids.add(pid)
                new_detection = True
        
        # Kirim HANYA jika ada orang baru
        if new_detection:
            g.last_global_send_time = now
            
            # Format Nama Unik (hilangkan duplikat)
            unique_names = list(set(detected_names))
            names_str = ", ".join(unique_names)
            
            print(f"🔔 ALERT TRIGGER: {names_str} (Cam {self.cam_id})")
            
            # Kompres Gambar (Quality 60%)
            _, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 60])
            
            # --- KIRIM WHATSAPP ---
            if CURRENT_CONFIG.get('waha_enabled'):
                b64 = base64.b64encode(buffer).decode('utf-8')
                # Caption Lengkap
                caption = (f"🚨 *SECURITY ALERT*\n"
                           f"📍 Cam: {self.cam_id}\n"
                           f"👥 Total: {len(current_ids)}\n"
                           f"📝 ID: *{names_str}*\n"
                           f"🕒 {time.strftime('%H:%M:%S')}")
                
                threading.Thread(target=send_whatsapp, args=(self.cam_id, caption, b64)).start()
            
            # --- KIRIM TELEGRAM ---
            if CURRENT_CONFIG.get('telegram_enabled'):
                img_bytes = buffer.tobytes()
                caption = (f"🚨 <b>SECURITY ALERT</b>\n"
                           f"📍 Cam: {self.cam_id}\n"
                           f"👥 Total: {len(current_ids)}\n"
                           f"📝 ID: <b>{names_str}</b>\n"
                           f"🕒 {time.strftime('%H:%M:%S')}")
                
                threading.Thread(target=send_telegram, args=(self.cam_id, caption, img_bytes)).start()

    def get_frame(self):
        """
        Mengambil frame untuk streaming ke Browser (Flask).
        Di-resize ke 640px max width agar ringan di jaringan.
        """
        with self.local_lock:
            if self.output_frame is None: return None
            
            h, w = self.output_frame.shape[:2]
            target_w = 640
            if w > target_w:
                scale = target_w / float(w)
                display = cv2.resize(self.output_frame, None, fx=scale, fy=scale)
            else:
                display = self.output_frame

            ret, buf = cv2.imencode(".jpg", display, [int(cv2.IMWRITE_JPEG_QUALITY), 50])
            return bytearray(buf) if ret else None
    
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
