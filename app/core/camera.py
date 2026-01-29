import os
# Paksa FFmpeg TCP agar stream stabil
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

import cv2
import time
import threading
import base64
import torch
import numpy as np
from ultralytics import YOLO

# --- IMPORT INTERNAL ---
# Perhatikan titik (.) berarti import dari folder yang sama (app/core)
from .globals import CURRENT_CONFIG, ACTIVE_STREAMS
from .plc import update_plc_status
from .notifier import send_whatsapp, send_telegram
from .face_engine import FaceEngine  # <--- IMPORT MODUL BARU KITA
import app.core.globals as g

# ==========================================
# 1. SETUP ENGINE (HYBRID)
# ==========================================

# A. Setup YOLO (GPU - GTX 745)
if torch.cuda.is_available():
    DEVICE = 'cuda:0'
    print(f"🚀 YOLO Engine: GPU {torch.cuda.get_device_name(0)}")
else:
    DEVICE = 'cpu'
    print("⚠️ YOLO Engine: CPU (Fallback)")

try:
    model = YOLO("yolov8n.pt")
    model.to(DEVICE)
except Exception as e:
    print(f"❌ Error Load YOLO: {e}")
    model = YOLO("yolov8n.pt")

# B. Setup Face Engine (CPU - i7 Gen 4)
# Folder 'known_faces' diasumsikan ada di root project
face_engine = FaceEngine("known_faces") 


# ==========================================
# 2. CLASS BUFFERLESS CAPTURE
# ==========================================
class BufferlessVideoCapture:
    def __init__(self, name):
        self.cap = cv2.VideoCapture(name)
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
# 3. CLASS CAM STREAM (HYBRID LOGIC)
# ==========================================
class CamStream(threading.Thread):
    def __init__(self, cam_id, source):
        threading.Thread.__init__(self)
        self.cam_id = cam_id
        self.source = int(source) if str(source).isdigit() else source
        self.running = True
        self.output_frame = None
        
        # Memori Notifikasi
        self.detected_ids = set() 
        self.last_detection_time = 0
        
        # --- MEMORI WAJAH ---
        self.face_cache = {}    # { id_yolo : "Nama" }
        self.frame_count = 0    
        self.rec_interval = 10  # Cek wajah tiap 10 frame
        
        self.local_lock = threading.Lock()
        self.cap = None 

    def run(self):
        print(f"🎥 Start Cam {self.cam_id} [Hybrid Mode]")
        self.cap = BufferlessVideoCapture(self.source)
        time.sleep(1)

        reset_tracker = True 
        
        while self.running:
            success, frame = self.cap.read()
            
            if not success or frame is None or frame.size == 0:
                print(f"⚠️ Cam {self.cam_id} Reconnecting...")
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

                # 1. YOLO TRACKING (GPU)
                # imgsz=640 (HD) agar crop wajah jelas
                results = model.track(
                    source=frame,
                    persist=not reset_tracker,
                    classes=[0],
                    conf=conf,
                    imgsz=640,
                    verbose=False,
                    device=DEVICE,
                    half=False,
                    stream=False
                )
                
                if reset_tracker: reset_tracker = False

                # 2. PROSES DETEKSI & WAJAH
                if results[0].boxes and results[0].boxes.id is not None:
                    has_person = True
                    
                    boxes = results[0].boxes.xyxy.cpu().numpy()
                    track_ids = results[0].boxes.id.cpu().numpy().astype(int)

                    for box, track_id in zip(boxes, track_ids):
                        x1, y1, x2, y2 = map(int, box)
                        
                        # --- LOGIKA HYBRID FACE REC ---
                        detected_name = self.face_cache.get(track_id, "Unknown")
                        
                        # Trigger Check:
                        # a. Interval tercapai (tiap 10 frame)
                        # b. ATAU ID ini benar-benar baru
                        is_time_check = (self.frame_count % self.rec_interval == 0)
                        is_new_id = (track_id not in self.face_cache)
                        box_h = y2 - y1

                        # Hanya cek wajah jika orangnya cukup dekat/besar (>100px)
                        if (is_time_check or is_new_id) and box_h > 100:
                            
                            # Crop Badan/Wajah dengan aman
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
                        
                        # --- GAMBAR VISUAL ---
                        # Hijau = Dikenal, Merah = Unknown
                        color = (0, 255, 0) if detected_name != "Unknown" else (0, 0, 255)
                        
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                        
                        # Label: ID + Nama
                        label = f"#{track_id} {detected_name}"
                        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                        cv2.rectangle(annotated_frame, (x1, y1 - 25), (x1 + w, y1), color, -1)
                        cv2.putText(annotated_frame, label, (x1, y1 - 5), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                    # Trigger Notif (Hanya jika person detected)
                    # Kita passing results[0] saja untuk logika notif standar
                    self.handle_alert(results[0], annotated_frame)

                # Update PLC
                update_plc_status(self.cam_id, has_person)
                
                with self.local_lock:
                    self.output_frame = annotated_frame 
            
            except Exception as e:
                # Handle error GPU OOM atau error lain
                if "out of memory" in str(e).lower():
                    torch.cuda.empty_cache()
                print(f"❌ Error Cam {self.cam_id}: {e}")
                reset_tracker = True
                with self.local_lock:
                    self.output_frame = frame

        if self.cap:
            self.cap.release()
        print(f"🛑 Cam {self.cam_id} Stopped")

    def handle_alert(self, result, frame):
        # (LOGIKA NOTIFIKASI SAMA SEPERTI SEBELUMNYA)
        # Salin logika handle_alert Anda di sini...
        pass 

    def get_frame(self):
        # (LOGIKA GET FRAME SAMA SEPERTI SEBELUMNYA)
        with self.local_lock:
            if self.output_frame is None: return None
            # Resize ke 640px max width
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

# --- FUNGSI RESTART (Tetap Sama) ---
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
