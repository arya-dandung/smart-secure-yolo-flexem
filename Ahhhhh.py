import os
# Paksa OpenCV menggunakan CPU untuk Video I/O agar tidak rebutan dengan DirectML
os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"] = "0"

import cv2
import time
import threading
import base64
import numpy as np
import app.core.globals as g

# Import InsightFace
from insightface.app import FaceAnalysis

# Import modul internal
from .globals import CURRENT_CONFIG, ACTIVE_STREAMS
from .plc import update_plc_status
from .notifier import send_whatsapp, send_telegram

# ==========================================
# 1. KONFIGURASI AI (DIRECTML)
# ==========================================
print("🚀 Menginisialisasi InsightFace (DirectML Mode)...")

# Gunakan DmlExecutionProvider (DirectX 12)
app_face = FaceAnalysis(name='buffalo_s', providers=['DmlExecutionProvider', 'CPUExecutionProvider'])
app_face.prepare(ctx_id=0, det_size=(640, 640))

FOLDER_DATABASE = "database_wajah" 
known_embeddings = [] 
known_names = []
SIMILARITY_THRESHOLD = 0.50 

# PENTING: Interval Deteksi
# AI hanya akan jalan setiap 5 frame sekali. 
# Ini MENCEGAH program crash saat web dibuka.
FRAME_SKIP_INTERVAL = 5 

# ==========================================
# 2. LOAD DATABASE
# ==========================================
def load_face_database():
    global known_embeddings, known_names
    base_dir = os.path.dirname(os.path.abspath(__file__)) 
    root_dir = os.path.dirname(base_dir) 
    db_path = os.path.join(root_dir, FOLDER_DATABASE)

    if not os.path.exists(db_path):
        os.makedirs(db_path)
        return

    print(f"📂 Memuat Database dari '{db_path}'...")
    known_embeddings = []
    known_names = []
    
    for filename in os.listdir(db_path):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            path = os.path.join(db_path, filename)
            try:
                img = cv2.imread(path)
                if img is None: continue
                faces = app_face.get(img)
                if len(faces) > 0:
                    known_embeddings.append(faces[0].embedding)
                    name = os.path.splitext(filename)[0].replace("_", " ").upper()
                    name = ''.join([i for i in name if not i.isdigit()]).strip()
                    known_names.append(name)
            except Exception:
                pass
    print(f"✨ Total Database: {len(known_names)} wajah.")

load_face_database()

# ==========================================
# 3. BUFFERLESS CAPTURE (Anti Lag)
# ==========================================
class BufferlessVideoCapture:
    def __init__(self, name):
        self.cap = cv2.VideoCapture(name)
        # Set resolusi rendah agar ringan
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
                with self.lock: self.status = False
                time.sleep(0.5)
                continue
            with self.lock:
                self.latest_frame = frame
                self.status = True

    def read(self):
        with self.lock:
            return self.status, (self.latest_frame.copy() if self.latest_frame is not None else None)

    def release(self):
        self.running = False
        self.t.join()
        self.cap.release()

# ==========================================
# 4. STREAM PROCESSOR (SAFE MODE)
# ==========================================
class CamStream(threading.Thread):
    def __init__(self, cam_id, source):
        threading.Thread.__init__(self)
        self.cam_id = cam_id
        self.source = int(source) if str(source).isdigit() else source
        self.running = True
        self.output_frame = None
        
        # Variabel untuk "Ingatan" AI (Caching)
        self.last_faces_cache = [] 
        self.frame_count = 0
        
        self.detected_names_buffer = set() 
        self.last_detection_time = 0
        self.local_lock = threading.Lock() # Kunci Pengaman Thread
        self.cap = None 

    def run(self):
        print(f"🎥 Start Cam {self.cam_id}...")
        self.cap = BufferlessVideoCapture(self.source)
        time.sleep(1) 

        while self.running:
            success, frame = self.cap.read()
            
            if not success or frame is None:
                time.sleep(1)
                continue
            
            # Resize frame jika terlalu besar (Wajib untuk GTX 745)
            # Semakin kecil gambar, semakin kecil kemungkinan crash
            h, w = frame.shape[:2]
            if w > 640:
                scale = 640 / w
                frame = cv2.resize(frame, None, fx=scale, fy=scale)

            # --- LOGIKA "AI JALAN SANTAI" ---
            self.frame_count += 1
            
            # Cek apakah saatnya menjalankan AI (misal: tiap 5 frame)
            if self.frame_count % FRAME_SKIP_INTERVAL == 0:
                try:
                    # Jalankan InsightFace
                    faces = app_face.get(frame)
                    
                    # Update Cache Hasil Deteksi
                    self.last_faces_cache = []
                    current_names_in_frame = []
                    
                    for face in faces:
                        box = face.bbox.astype(int)
                        live_emb = face.embedding
                        
                        # Pencocokan Wajah
                        name = "UNKNOWN"
                        max_score = 0
                        if len(known_embeddings) > 0:
                            scores = np.dot(known_embeddings, live_emb) / (
                                np.linalg.norm(known_embeddings, axis=1) * np.linalg.norm(live_emb)
                            )
                            max_idx = np.argmax(scores)
                            max_score = scores[max_idx]
                            
                            if max_score >= SIMILARITY_THRESHOLD:
                                name = known_names[max_idx]
                        
                        # Simpan ke cache untuk digambar di frame-frame berikutnya
                        self.last_faces_cache.append({
                            "box": box,
                            "name": name,
                            "score": max_score
                        })
                        
                        if name != "UNKNOWN":
                            current_names_in_frame.append(name)
                    
                    # Logika PLC & Notifikasi (Hanya saat AI jalan)
                    update_plc_status(self.cam_id, len(faces) > 0)
                    self.handle_alert(current_names_in_frame, frame)
                    
                except Exception as e:
                    print(f"⚠️ AI Error (DirectML Glitch): {e}")
                    # Jika error, biarkan lewat (jangan stop program)
            
            # --- GAMBAR HASIL (Dari Cache) ---
            # Kita menggambar kotak berdasarkan ingatan terakhir AI
            # Jadi meskipun AI tidak jalan di frame ini, kotak tetap muncul (smooth)
            annotated_frame = frame.copy()
            for item in self.last_faces_cache:
                x1, y1, x2, y2 = item["box"]
                name = item["name"]
                score = item["score"]
                
                color = (0, 255, 0) if name != "UNKNOWN" else (0, 0, 255)
                label = f"{name} ({score:.2f})"
                
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                cv2.rectangle(annotated_frame, (x1, y1 - 20), (x1 + 150, y1), color, -1)
                cv2.putText(annotated_frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # Update Frame Output untuk Web
            with self.local_lock:
                self.output_frame = annotated_frame

        if self.cap: self.cap.release()

    def handle_alert(self, current_names, frame):
        now = time.time()
        cooldown = int(CURRENT_CONFIG.get('cooldown', 30))
        if (now - self.last_detection_time > cooldown):
            self.detected_names_buffer.clear()
        self.last_detection_time = now

        new_names = [n for n in current_names if n not in self.detected_names_buffer]
        if new_names:
            for n in new_names: self.detected_names_buffer.add(n)
            
            if (now - g.last_global_send_time > 5):
                g.last_global_send_time = now
                print(f"🔔 Notif: {new_names}")
                _, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 50])
                b64 = base64.b64encode(buffer).decode('utf-8')
                
                if CURRENT_CONFIG.get('waha_enabled'):
                    threading.Thread(target=send_whatsapp, args=(self.cam_id, len(current_names), b64)).start()

    def get_frame(self):
        # Fungsi ini dipanggil oleh Web Browser
        with self.local_lock:
            if self.output_frame is None: return None
            
            try:
                # Kompresi gambar menjadi JPEG untuk dikirim ke web
                ret, encoded = cv2.imencode(".jpg", self.output_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 40])
                return bytearray(encoded) if ret else None
            except Exception as e:
                print(f"Web Stream Error: {e}")
                return None
    
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
