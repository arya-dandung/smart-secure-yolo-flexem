import os
import cv2
import time
import threading
import base64
import numpy as np
import face_recognition
import torch 
from ultralytics import YOLO

# [PENTING] Paksa FFmpeg menggunakan TCP
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

# Import modul internal
from .globals import CURRENT_CONFIG, ACTIVE_STREAMS
from .plc import update_plc_status
from .notifier import send_whatsapp, send_telegram
import app.core.globals as g

# ==========================================
# KONFIGURASI
# ==========================================
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"🚀 [SYSTEM] Inference Device: {DEVICE.upper()}")

# Load Model
# Gunakan 'yolov8m.pt' jika GPU kuat, atau 'yolov8s.pt' jika ingin ringan
try:
    model = YOLO("yolov8s.pt").to(DEVICE)
except:
    model = YOLO("yolov8n.pt").to(DEVICE)

# --- TUNING PARAMETER ---
YOLO_CONF_TRACK = 0.25      # 25% Yakin = Tampilkan Kotak (Sangat Sensitif)
FACE_TOLERANCE = 0.54       # Toleransi kemiripan wajah
INTERVAL_UNKNOWN = 5        # Cek wajah tiap 5 frame jika Unknown
INTERVAL_KNOWN = 30         # Cek wajah tiap 30 frame jika Kenal

FOLDER_DATABASE = "database_wajah"
known_face_encodings = []
known_face_names = []

# ==========================================
# HELPER FUNCTIONS
# ==========================================
def sanitize_image_for_dlib(img_cv):
    if img_cv is None: return None
    return np.ascontiguousarray(img_cv[:, :, ::-1])

def load_face_database():
    global known_face_encodings, known_face_names
    base_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(base_dir)
    db_path = os.path.join(root_dir, FOLDER_DATABASE)

    if not os.path.exists(db_path):
        os.makedirs(db_path)
        return

    print(f"📂 [LOAD] Database Wajah dari '{db_path}'...")
    known_face_encodings = []
    known_face_names = []
    
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
            except Exception: pass     
    print(f"✨ [READY] Total Database: {len(known_face_encodings)} wajah.")

load_face_database()

# ==========================================
# CLASS 1: BUFFERLESS CAPTURE (HD VERSION)
# ==========================================
class BufferlessVideoCapture:
    def __init__(self, name):
        self.cap = cv2.VideoCapture(name)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        # USAHAKAN RESOLUSI HD AGAR WAJAH JELAS
        if isinstance(name, int):
            self.cap.set(3, 1280) 
            self.cap.set(4, 720)
            
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
                with self.lock: self.status = False
                time.sleep(0.5)
                continue
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
# CLASS 2: HYBRID STREAM (LOGIKA TERPISAH)
# ==========================================
class CamStream(threading.Thread):
    def __init__(self, cam_id, source):
        threading.Thread.__init__(self)
        self.cam_id = cam_id
        self.source = int(source) if str(source).isdigit() else source
        self.running = True
        self.output_frame = None
        self.local_lock = threading.Lock()
        
        self.detected_ids = set()
        self.last_detection_time = 0
        self.cap = None 
        
        # DATABASE MEMORI SEMENTARA
        # track_history menyimpan: { ID_YOLO : "Nama Orang" }
        self.track_history = {} 
        self.track_timers = {} 

    def run(self):
        print(f"🎥 [START] Cam {self.cam_id} using {DEVICE}...")
        self.cap = BufferlessVideoCapture(self.source)
        time.sleep(1)

        while self.running:
            success, frame = self.cap.read()
            if not success or frame is None:
                time.sleep(0.1)
                continue
            
            try:
                # -----------------------------------------------------------
                # 1. YOLO TRACKING (MASTER POSISI)
                # -----------------------------------------------------------
                results = model.track(frame, persist=True, classes=[0], 
                                    tracker="bytetrack.yaml",
                                    conf=YOLO_CONF_TRACK, verbose=False, device=DEVICE)
                
                boxes = []
                track_ids = []
                
                if results[0].boxes.id is not None:
                    boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
                    track_ids = results[0].boxes.id.cpu().numpy().astype(int)
                
                has_person = len(boxes) > 0
                update_plc_status(self.cam_id, has_person)

                annotated_frame = frame.copy()
                current_frame_ids = []

                if has_person:
                    for box, track_id in zip(boxes, track_ids):
                        x1, y1, x2, y2 = box
                        current_frame_ids.append(track_id)
                        
                        # Pastikan ID ada di memory, Default = "Unknown"
                        if track_id not in self.track_history:
                            self.track_history[track_id] = "Unknown"
                        if track_id not in self.track_timers:
                            self.track_timers[track_id] = 0

                        # -------------------------------------------------------
                        # 2. FACE RECOGNITION (WORKER BACKGROUND)
                        # Hanya bertugas mengupdate text di self.track_history
                        # Tidak mempengaruhi kotak.
                        # -------------------------------------------------------
                        if self.track_timers[track_id] <= 0:
                            # CROP FULL BODY + PADDING (Agar wajah nunduk kena)
                            h_img, w_img = frame.shape[:2]
                            pad_y = int((y2 - y1) * 0.1)
                            pad_x = int((x2 - x1) * 0.1)
                            
                            crop_y1 = max(0, y1 - pad_y)
                            crop_y2 = min(h_img, y2 + pad_y)
                            crop_x1 = max(0, x1 - pad_x)
                            crop_x2 = min(w_img, x2 + pad_x)
                            
                            crop_img = frame[crop_y1:crop_y2, crop_x1:crop_x2]
                            
                            found_name = "Unknown"
                            
                            if crop_img.size > 0:
                                rgb_crop = sanitize_image_for_dlib(crop_img)
                                # Gunakan HOG agar aman di GTX 745
                                # upsample=1 agar wajah kecil terbaca
                                try:
                                    face_locs = face_recognition.face_locations(rgb_crop, number_of_times_to_upsample=1, model="hog")
                                    
                                    if face_locs:
                                        face_enc = face_recognition.face_encodings(rgb_crop, face_locs)[0]
                                        if len(known_face_encodings) > 0:
                                            matches = face_recognition.compare_faces(known_face_encodings, face_enc, tolerance=FACE_TOLERANCE)
                                            dists = face_recognition.face_distance(known_face_encodings, face_enc)
                                            if True in matches:
                                                found_name = known_face_names[np.argmin(dists)]
                                                print(f"      ✅ Match ID {track_id}: {found_name}")
                                except:
                                    pass # Jika error/VRAM full, skip frame ini, coba lagi nanti

                            # UPDATE LABEL
                            # Jika sebelumnya Unknown dan sekarang ketemu, update!
                            # Jika sebelumnya sudah Kenal, pertahankan (kecuali ganti orang)
                            self.track_history[track_id] = found_name
                            
                            # Reset Timer
                            if found_name != "Unknown":
                                self.track_timers[track_id] = INTERVAL_KNOWN
                            else:
                                self.track_timers[track_id] = INTERVAL_UNKNOWN
                        else:
                            self.track_timers[track_id] -= 1

                        # -------------------------------------------------------
                        # 3. VISUALISASI (BERGANTUNG PADA YOLO)
                        # Kotak digambar APAPUN hasil Face Rec-nya
                        # -------------------------------------------------------
                        display_name = self.track_history.get(track_id, "Unknown")
                        
                        # Warna: Hijau jika kenal, Merah jika Unknown
                        color = (0, 255, 0) if display_name != "Unknown" else (0, 0, 255)
                        
                        # Gambar Kotak (Dari data YOLO)
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                        
                        # Gambar Label
                        label = f"{display_name} [{track_id}]"
                        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                        cv2.rectangle(annotated_frame, (x1, y1 - 25), (x1 + t_size[0] + 10, y1), color, -1)
                        cv2.putText(annotated_frame, label, (x1 + 5, y1 - 5), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                    # Kirim Notifikasi
                    self.handle_alert(track_ids, annotated_frame)
                
                # Garbage Collection
                if len(self.track_history) > 20:
                    active_set = set(current_frame_ids)
                    self.track_history = {k:v for k,v in self.track_history.items() if k in active_set}
                    self.track_timers = {k:v for k,v in self.track_timers.items() if k in active_set}

                with self.local_lock:
                    self.output_frame = annotated_frame.copy()
            
            except Exception as e:
                print(f"❌ [ERROR] Cam {self.cam_id}: {e}")
                with self.local_lock: self.output_frame = frame

        if self.cap: self.cap.release()

    def handle_alert(self, current_ids, frame):
        now = time.time()
        cooldown = int(CURRENT_CONFIG.get('cooldown', 30))
        
        if (now - self.last_detection_time > cooldown):
            self.detected_ids.clear()
        
        self.last_detection_time = now
        if len(current_ids) == 0: return
        
        if (now - g.last_global_send_time > cooldown):
            new_person = False
            for pid in current_ids:
                if pid not in self.detected_ids:
                    self.detected_ids.add(pid)
                    new_person = True
            
            if new_person:
                g.last_global_send_time = now
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
            h, w = self.output_frame.shape[:2]
            if w > 800:
                scale = 800 / float(w)
                out = cv2.resize(self.output_frame, None, fx=scale, fy=scale)
            else:
                out = self.output_frame
            ret, encoded = cv2.imencode(".jpg", out, [int(cv2.IMWRITE_JPEG_QUALITY), 60])
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
