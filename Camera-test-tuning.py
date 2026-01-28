 
import os
# Paksa FFmpeg TCP agar RTSP stabil
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

import cv2
import time
import threading
import base64
import numpy as np
import app.core.globals as g

# Import InsightFace
from insightface.app import FaceAnalysis

# Import modul internal project Anda
from .globals import CURRENT_CONFIG, ACTIVE_STREAMS
from .plc import update_plc_status
from .notifier import send_whatsapp, send_telegram

# ==========================================
# 1. KONFIGURASI GLOBAL & MODEL (INSIGHTFACE)
# ==========================================
print("🚀 Menginisialisasi InsightFace pada GPU...")

# 'buffalo_l' adalah model paling akurat & cepat dari InsightFace
# providers=['CUDAExecutionProvider'] MEMAKSA pakai NVIDIA GPU
app_face = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])

# det_size=(640, 640): Resolusi deteksi.
# Jika wajah kecil tidak terdeteksi, ubah jadi (320, 320) atau (1280, 1280)
app_face.prepare(ctx_id=0, det_size=(640, 640))

FOLDER_DATABASE = "database_wajah" 
known_embeddings = [] # Pengganti 'encodings'
known_names = []

# Threshold Kemiripan (0.0 - 1.0)
# Di InsightFace, 0.4 sampai 0.6 adalah angka ideal.
# Semakin TINGGI = Semakin KETAT.
SIMILARITY_THRESHOLD = 0.50 

# ==========================================
# 2. LOAD DATABASE WAJAH
# ==========================================
def load_face_database():
    global known_embeddings, known_names
    
    base_dir = os.path.dirname(os.path.abspath(__file__)) 
    root_dir = os.path.dirname(base_dir) 
    db_path = os.path.join(root_dir, FOLDER_DATABASE)

    if not os.path.exists(db_path):
        os.makedirs(db_path)
        print(f"📂 [INFO] Folder database dibuat di: {db_path}")
        return

    print(f"📂 Memuat Database InsightFace dari '{db_path}'...")
    known_embeddings = []
    known_names = []
    
    count = 0
    for filename in os.listdir(db_path):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            path = os.path.join(db_path, filename)
            try:
                img = cv2.imread(path)
                if img is None: continue

                # Deteksi wajah di foto database
                faces = app_face.get(img)
                
                if len(faces) > 0:
                    # Ambil wajah terbesar (biasanya index 0 sorted by size)
                    # Simpan 'embedding' (kode unik wajah)
                    known_embeddings.append(faces[0].embedding)
                    
                    # Bersihkan nama file
                    name = os.path.splitext(filename)[0].replace("_", " ").upper()
                    name = ''.join([i for i in name if not i.isdigit()]).strip()
                    known_names.append(name)
                    
                    count += 1
                    print(f"   ✅ OK: {name}")
                else:
                    print(f"   ⚠️ Wajah tidak ditemukan di file: {filename}")

            except Exception as e:
                print(f"   ❌ Error {filename}: {e}")
                
    print(f"✨ Total Database: {count} wajah siap.")

load_face_database()

# ==========================================
# 3. BUFFERLESS CAPTURE (Tetap Sama)
# ==========================================
class BufferlessVideoCapture:
    def __init__(self, name):
        self.cap = cv2.VideoCapture(name)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1) # Anti Lag
        
        if isinstance(name, int):
            self.cap.set(3, 640)
            self.cap.set(4, 480)
            
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
# 4. STREAM PROCESSOR (MODERN ENGINE)
# ==========================================
class CamStream(threading.Thread):
    def __init__(self, cam_id, source):
        threading.Thread.__init__(self)
        self.cam_id = cam_id
        self.source = int(source) if str(source).isdigit() else source
        self.running = True
        self.output_frame = None
        
        # Logika Notifikasi
        self.detected_names_buffer = set() 
        self.last_detection_time = 0
        self.local_lock = threading.Lock()
        self.cap = None 

    def run(self):
        print(f"🎥 Start Cam {self.cam_id} (InsightFace Engine)...")
        self.cap = BufferlessVideoCapture(self.source)
        time.sleep(1) 

        while self.running:
            success, frame = self.cap.read()
            
            if not success or frame is None:
                # Reconnect logic
                print(f"⚠️ Cam {self.cam_id} Reconnecting...")
                time.sleep(2)
                self.cap.release()
                self.cap = BufferlessVideoCapture(self.source)
                time.sleep(1)
                continue
            
            try:
                # --- PROSES AI (INSIGHTFACE) ---
                # app_face.get(frame) melakukan 2 hal sekaligus:
                # 1. Deteksi lokasi wajah (Face Detection)
                # 2. Ekstrak kode wajah (Face Embedding)
                # Ini berjalan di GPU, jadi sangat cepat.
                faces = app_face.get(frame)
                
                has_person = len(faces) > 0
                update_plc_status(self.cam_id, has_person)

                annotated_frame = frame.copy()
                current_names_in_frame = []

                if has_person:
                    for face in faces:
                        # Ambil koordinat kotak (BBox)
                        # InsightFace mengembalikan float, harus jadi int
                        box = face.bbox.astype(int)
                        x1, y1, x2, y2 = box[0], box[1], box[2], box[3]

                        # Ambil Embedding Wajah Ini
                        live_emb = face.embedding
                        
                        # --- LOGIKA PENCOCOKAN (COSINE SIMILARITY) ---
                        max_score = 0
                        name = "UNKNOWN"
                        
                        if len(known_embeddings) > 0:
                            for idx, db_emb in enumerate(known_embeddings):
                                # Hitung kemiripan (Rumus Matematika Vektor)
                                # Hasilnya antara -1 sampai 1. (1 = Mirip Sempurna)
                                score = np.dot(live_emb, db_emb) / (np.linalg.norm(live_emb) * np.linalg.norm(db_emb))
                                
                                if score > max_score:
                                    max_score = score
                                    if score >= SIMILARITY_THRESHOLD:
                                        name = known_names[idx]
                        
                        # Simpan nama untuk notifikasi
                        if name != "UNKNOWN":
                            current_names_in_frame.append(name)

                        # --- VISUALISASI ---
                        # Hijau jika kenal, Merah jika Unknown
                        color = (0, 255, 0) if name != "UNKNOWN" else (0, 0, 255)
                        
                        # Tampilkan Score kemiripan untuk debug (misal: ARYA 0.65)
                        label = f"{name} ({max_score:.2f})"
                        
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                        
                        # Background Label
                        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                        cv2.rectangle(annotated_frame, (x1, y1 - 25), (x1 + w, y1), color, -1)
                        cv2.putText(annotated_frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                    # Trigger Notifikasi
                    self.handle_alert(current_names_in_frame, annotated_frame)
                
                # Update Output Frame
                with self.local_lock:
                    self.output_frame = annotated_frame.copy()
            
            except Exception as e:
                print(f"❌ Error Logic: {e}")
                with self.local_lock:
                    self.output_frame = frame

        if self.cap: self.cap.release()
        print(f"🛑 Cam {self.cam_id} Stopped")

    def handle_alert(self, current_names, frame):
        """
        Logika Notifikasi Berbasis Nama (Bukan ID Tracking).
        Jika "ARYA" muncul, kirim notif. Lalu cooldown 30 detik untuk "ARYA".
        Jika "BOS" muncul, kirim notif "BOS" (meskipun Arya sedang cooldown).
        """
        now = time.time()
        cooldown = int(CURRENT_CONFIG.get('cooldown', 30))
        
        # Reset buffer nama jika sudah lama sepi
        if (now - self.last_detection_time > cooldown):
            self.detected_names_buffer.clear()
        self.last_detection_time = now

        if not current_names: return

        # Cek apakah ada nama BARU yang belum dikirim notifnya
        new_names_found = []
        for name in current_names:
            if name not in self.detected_names_buffer:
                new_names_found.append(name)
                self.detected_names_buffer.add(name) # Masukkan ke daftar "sudah dikirim"

        # Jika ada orang baru ATAU Global Cooldown sudah lewat (untuk reminder)
        # Kita gunakan logika sederhana: Kirim jika ada nama baru terdeteksi
        if new_names_found:
            # Cek Global Cooldown (agar tidak spamming WA per detik)
            if (now - g.last_global_send_time > 5): # Minimal jarak antar WA 5 detik
                g.last_global_send_time = now
                
                print(f"🔔 Notif Triggered: {new_names_found}")
                
                # Compress gambar
                _, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 60])
                count = len(current_names)

                if CURRENT_CONFIG.get('waha_enabled'):
                    b64 = base64.b64encode(buffer).decode('utf-8')
                    threading.Thread(target=send_whatsapp, args=(self.cam_id, count, b64)).start()
                
                if CURRENT_CONFIG.get('telegram_enabled'):
                    img_bytes = buffer.tobytes()
                    threading.Thread(target=send_telegram, args=(self.cam_id, count, img_bytes)).start()

    def get_frame(self):
        with self.local_lock:
            if self.output_frame is None: return None
            # Resize agar Web Interface ringan
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
