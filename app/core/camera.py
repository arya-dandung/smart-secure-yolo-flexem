import os
# [PENTING] Paksa FFmpeg menggunakan TCP untuk RTSP agar gambar tidak rusak/abu-abu
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

import cv2
import time
import threading
import base64
import queue
from ultralytics import YOLO

# Import modul internal
from .globals import CURRENT_CONFIG, ACTIVE_STREAMS
from .plc import update_plc_status
from .notifier import send_whatsapp, send_telegram
import app.core.globals as g

# Load model di global scope agar hemat memori
model = YOLO("yolov8n.pt")

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
            return self.status, self.latest_frame

    def release(self):
        self.running = False
        self.t.join()
        self.cap.release()

# ==========================================
# CLASS 2: CAMERA STREAM PROCESSOR
# ==========================================
class CamStream(threading.Thread):
    def __init__(self, cam_id, source):
        threading.Thread.__init__(self)
        self.cam_id = cam_id
        self.source = int(source) if str(source).isdigit() else source
        self.running = True
        self.output_frame = None
        
        # Set untuk memori ID notifikasi (agar tidak spam WA)
        self.detected_ids = set() 
        
        # [BARU] Menyimpan waktu terakhir kali ada orang terdeteksi
        # Digunakan untuk mereset memori jika scene kosong lama
        self.last_detection_time = 0
        
        self.local_lock = threading.Lock()
        self.cap = None 

    def run(self):
        print(f"🎥 Start Cam {self.cam_id} on {self.source}")
        
        self.cap = BufferlessVideoCapture(self.source)
        time.sleep(1) # Warmup

        reset_tracker = True 

        while self.running:
            success, frame = self.cap.read()
            
            # 1. Validasi Frame
            if not success or frame is None or frame.size == 0:
                print(f"⚠️ Cam {self.cam_id} Reconnecting...")
                time.sleep(2)
                self.cap.release()
                self.cap = BufferlessVideoCapture(self.source)
                reset_tracker = True 
                time.sleep(1)
                continue
            
            try:
                conf = float(CURRENT_CONFIG.get('confidence', 0.5))
                
                # 2. YOLO Tracking
                if reset_tracker:
                    results = model.track(frame, persist=False, classes=[0], conf=conf, verbose=False)
                    reset_tracker = False
                else:
                    results = model.track(frame, persist=True, classes=[0], conf=conf, verbose=False)

                # 3. Cek Keberadaan Orang
                has_person = False
                if results[0].boxes and results[0].boxes.id is not None:
                    has_person = True

                # === [LOGIKA PLC] ===
                update_plc_status(self.cam_id, has_person)

                # 4. Gambar Anotasi & Handle Notifikasi
                annotated_frame = results[0].plot(line_width=2, font_size=1, conf=False, img=frame)
                
                if has_person:
                    self.handle_alert(results[0], annotated_frame)
                
                # 5. Update Frame untuk Web View
                with self.local_lock:
                    self.output_frame = annotated_frame.copy()
            
            except Exception as e:
                print(f"❌ Error Cam {self.cam_id}: {e}")
                reset_tracker = True
                with self.local_lock:
                    self.output_frame = frame

        if self.cap:
            self.cap.release()
        print(f"🛑 Cam {self.cam_id} Stopped")

    def handle_alert(self, result, frame):
        """
        Menangani pengiriman Notifikasi (WA/Telegram).
        Menggunakan Cooldown agar tidak spam, tapi tetap mengingat orang lama.
        """
        now = time.time()
        cooldown = int(CURRENT_CONFIG.get('cooldown', 30))
        
        # [LOGIKA BARU]
        # Jangan reset memori berdasarkan waktu kirim terakhir (last_global_send_time).
        # Tapi reset HANYA jika sudah lama tidak ada deteksi (misal 30 detik kosong).
        if (now - self.last_detection_time > cooldown):
            self.detected_ids.clear()
        
        # Update waktu terakhir deteksi (karena fungsi ini dipanggil saat ada orang)
        self.last_detection_time = now

        # Ambil ID yang terdeteksi
        if result.boxes.id is None: return
        current_ids = result.boxes.id.cpu().numpy().astype(int)
        
        new_detection = False
        
        # Cek Global Cooldown (Hanya untuk membatasi frekuensi kirim WA, bukan reset memori)
        if (now - g.last_global_send_time > cooldown):
            for pid in current_ids:
                if pid not in self.detected_ids:
                    self.detected_ids.add(pid)
                    new_detection = True
            
            if new_detection:
                g.last_global_send_time = now
                print(f"🔔 Notif Triggered by Cam {self.cam_id}")
                
                count = len(current_ids)
                
                # Encode ringan untuk notif (Quality 50%)
                _, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 50])
                
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
            
            # Optimasi Resize (Max width 640px)
            height, width = self.output_frame.shape[:2]
            target_width = 640
            
            if width > target_width:
                scale = target_width / float(width)
                display_frame = cv2.resize(self.output_frame, None, fx=scale, fy=scale)
            else:
                display_frame = self.output_frame

            # Optimasi Compress JPEG (Quality 50)
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