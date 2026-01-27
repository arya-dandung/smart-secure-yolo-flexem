# import cv2
# import time
# import threading
# import base64
# from ultralytics import YOLO
# from .globals import CURRENT_CONFIG, ACTIVE_STREAMS, lock
# from .plc import trigger_plc
# from .notifier import send_whatsapp, send_telegram
# import app.core.globals as g

# # Load model di global scope
# model = YOLO("yolov8n.pt")

# class CamStream(threading.Thread):
#     def __init__(self, cam_id, source):
#         threading.Thread.__init__(self)
#         self.cam_id = cam_id
#         # Parsing source: jika angka (0,1) jadikan int, jika RTSP string biarkan
#         self.source = int(source) if str(source).isdigit() else source
#         self.running = True
#         self.output_frame = None
#         self.detected_ids = set()
#         self.local_lock = threading.Lock()

#     def run(self):
#         print(f"🎥 Start Cam {self.cam_id} on {self.source}")
#         cap = cv2.VideoCapture(self.source)
        
#         # Optimize Webcam
#         if isinstance(self.source, int):
#             cap.set(3, 640)
#             cap.set(4, 480)

#         while self.running:
#             success, frame = cap.read()
#             if not success:
#                 time.sleep(2)
#                 cap.release()
#                 cap = cv2.VideoCapture(self.source)
#                 continue
            
#             try:
#                 conf = float(CURRENT_CONFIG.get('confidence', 0.5))
#                 # YOLO Process
#                 results = model.track(frame, persist=True, classes=[0], conf=conf, verbose=False)
#                 annotated_frame = results[0].plot(line_width=2, font_size=1, conf=False, img=frame)

#                 if results[0].boxes.id is not None:
#                     self.handle_alert(results[0], annotated_frame)
                
#                 with self.local_lock:
#                     self.output_frame = annotated_frame.copy()
#             except Exception as e:
#                 print(f"Error Cam {self.cam_id}: {e}")
#                 with self.local_lock:
#                     self.output_frame = frame

#         cap.release()
#         print(f"🛑 Cam {self.cam_id} Stopped")

#     def handle_alert(self, result, frame):
#         now = time.time()
#         cooldown = int(CURRENT_CONFIG.get('cooldown', 30))
        
#         current_ids = result.boxes.id.cpu().numpy().astype(int)
#         new_detection = False
        
#         # Cek Global Cooldown (Anti Spam Antar Kamera)
#         if (now - g.last_global_send_time > cooldown):
#             for pid in current_ids:
#                 if pid not in self.detected_ids:
#                     self.detected_ids.add(pid)
#                     new_detection = True
            
#             if new_detection:
#                 g.last_global_send_time = now
#                 print(f"🔔 Alert Triggered by Cam {self.cam_id}")
                
#                 # 1. Trigger PLC (Async)
#                 threading.Thread(target=trigger_plc, args=(self.cam_id,)).start()
                
#                 # 2. Encode Image
#                 # _, buffer = cv2.imencode('.jpg', frame)
#                 _, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 50])

#                 # 3. Notifiers (Async)
#                 count = len(current_ids)
#                 if CURRENT_CONFIG.get('waha_enabled'):
#                     b64 = base64.b64encode(buffer).decode('utf-8')
#                     threading.Thread(target=send_whatsapp, args=(self.cam_id, count, b64)).start()
                
#                 if CURRENT_CONFIG.get('telegram_enabled'):
#                     img_bytes = buffer.tobytes()
#                     threading.Thread(target=send_telegram, args=(self.cam_id, count, img_bytes)).start()

#     def get_frame(self):
#     #Before optimalisasi
#         # with self.local_lock:
#         #     if self.output_frame is None: return None
#         #     ret, encoded = cv2.imencode(".jpg", self.output_frame)
#         #     return bytearray(encoded) if ret else None

#     #After optimalisasi
#         with self.local_lock:
#             if self.output_frame is None: return None

#             # --- OPTIMASI 1: RESIZE (Kecilkan Resolusi) ---
#             # Mengubah ukuran menjadi lebar 640px (tinggi menyesuaikan rasio)
#             # Ini sangat efektif jika kamera asli Anda 1080p atau lebih
#             height, width = self.output_frame.shape[:2]
#             target_width = 640
#             if width > target_width:
#                 scaling_factor = target_width / float(width)
#                 display_frame = cv2.resize(self.output_frame, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)
#             else:
#                 display_frame = self.output_frame

#         # --- OPTIMASI 2: KOMPRESI JPEG (Turunkan Kualitas) ---
#         # Angka 50 adalah kualitas (0-100). Default OpenCV biasanya 95.
#         # Menurunkan ke 40-60 mengurangi ukuran file drastis tanpa terlihat buram di HP/Laptop.
#         encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 50]
        
#         ret, encoded = cv2.imencode(".jpg", display_frame, encode_param)
#         return bytearray(encoded) if ret else None
    
#     def stop(self):
#         self.running = False
#         self.join()

# def restart_camera_threads():
#     # Stop existing
#     for stream in ACTIVE_STREAMS.values():
#         stream.stop()
#     ACTIVE_STREAMS.clear()
    
#     # Start new based on config
#     cams = CURRENT_CONFIG.get('cameras', {})
#     for cid, src in cams.items():
#         if src and str(src).strip():
#             stream = CamStream(cid, src)
#             stream.daemon = True
#             stream.start()
#             ACTIVE_STREAMS[cid] = stream

import os
# [PENTING] Paksa FFmpeg menggunakan TCP untuk RTSP agar gambar tidak rusak/abu-abu
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

import cv2
import time
import threading
import base64
import queue
from ultralytics import YOLO
from .globals import CURRENT_CONFIG, ACTIVE_STREAMS
from .plc import trigger_plc
from .notifier import send_whatsapp, send_telegram
import app.core.globals as g

# Load model di global scope agar tidak reload berulang kali
# Pastikan file yolov8n.pt ada di folder root proyek
model = YOLO("yolov8n.pt")

# ==========================================
# CLASS 1: BUFFERLESS VIDEO CAPTURE
# Bertugas menyedot frame secepat mungkin & membuang antrian lama
# ==========================================
class BufferlessVideoCapture:
    def __init__(self, name):
        self.cap = cv2.VideoCapture(name)
        
        # Optimize Webcam (Jika input berupa angka/USB Cam)
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
                # Jika koneksi putus, beri sinyal false
                with self.lock:
                    self.status = False
                time.sleep(0.5) # Tunggu sebentar sebelum retry internal
                continue
            
            # Simpan HANYA frame terbaru, frame lama tertimpa
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
# Mengambil frame dari Bufferless -> Deteksi YOLO -> Kirim Alert -> Siapkan untuk Web
# ==========================================
class CamStream(threading.Thread):
    def __init__(self, cam_id, source):
        threading.Thread.__init__(self)
        self.cam_id = cam_id
        # Parsing source: jika angka (0,1) jadikan int, jika RTSP string biarkan
        self.source = int(source) if str(source).isdigit() else source
        self.running = True
        self.output_frame = None
        self.detected_ids = set()
        self.local_lock = threading.Lock()
        self.cap = None 

    def run(self):
        print(f"🎥 Start Cam {self.cam_id} on {self.source}")
        
        # Inisialisasi Bufferless Capture
        self.cap = BufferlessVideoCapture(self.source)
        time.sleep(1) # Pemanasan kamera

        # Flag untuk mereset Tracker jika stream baru mulai/restart
        # Ini SOLUSI untuk error 'lkpyramid' / 'Assertion failed'
        reset_tracker = True 

        while self.running:
            # Ambil frame terbaru (instan, tidak menunggu buffer)
            success, frame = self.cap.read()
            
            # Validasi Frame: Pastikan sukses DAN frame punya isi (tidak kosong)
            if not success or frame is None or frame.size == 0:
                print(f"⚠️ Cam {self.cam_id} No Signal / Reconnecting...")
                time.sleep(2)
                
                # Re-init capture jika putus total
                self.cap.release()
                self.cap = BufferlessVideoCapture(self.source)
                
                # Tandai agar tracker di-reset saat nyambung lagi
                reset_tracker = True 
                time.sleep(1)
                continue
            
            try:
                conf = float(CURRENT_CONFIG.get('confidence', 0.5))
                
                # --- LOGIKA ANTI-CRASH YOLO ---
                # Jika ini frame pertama setelah restart/error, persist=False (Lupakan sejarah)
                # Jika frame lancar, persist=True (Lanjutkan tracking ID)
                if reset_tracker:
                    results = model.track(frame, persist=False, classes=[0], conf=conf, verbose=False)
                    reset_tracker = False # Selanjutnya kembali normal
                else:
                    results = model.track(frame, persist=True, classes=[0], conf=conf, verbose=False)

                # Gambar kotak deteksi di frame
                annotated_frame = results[0].plot(line_width=2, font_size=1, conf=False, img=frame)

                # Jika ada deteksi, proses alert
                if results[0].boxes.id is not None:
                    self.handle_alert(results[0], annotated_frame)
                
                # Update frame untuk ditampilkan di Web
                with self.local_lock:
                    self.output_frame = annotated_frame.copy()
            
            except Exception as e:
                # Tangkap error spesifik agar thread tidak mati total
                print(f"❌ Error Cam {self.cam_id}: {e}")
                
                # Jika error terjadi (misal lkpyramid), paksa reset tracker di putaran berikutnya
                reset_tracker = True
                
                # Tetap tampilkan frame polos agar user tahu kamera masih hidup
                with self.local_lock:
                    self.output_frame = frame

        # Cleanup saat thread berhenti
        if self.cap:
            self.cap.release()
        print(f"🛑 Cam {self.cam_id} Stopped")

    def handle_alert(self, result, frame):
        """
        Fungsi ini menangani logika Cooldown & Pengiriman Notifikasi
        """
        now = time.time()
        cooldown = int(CURRENT_CONFIG.get('cooldown', 30))
        
        # Ambil ID unik setiap orang yang terdeteksi
        current_ids = result.boxes.id.cpu().numpy().astype(int)
        new_detection = False
        
        # Cek Global Cooldown (Anti Spam Notifikasi)
        if (now - g.last_global_send_time > cooldown):
            for pid in current_ids:
                if pid not in self.detected_ids:
                    self.detected_ids.add(pid)
                    new_detection = True
            
            if new_detection:
                g.last_global_send_time = now
                print(f"🔔 Alert Triggered by Cam {self.cam_id}")
                
                # 1. Trigger PLC (Jalan di background thread)
                threading.Thread(target=trigger_plc, args=(self.cam_id,)).start()
                
                # 2. Encode Image untuk dikirim ke WA/Telegram
                # Kualitas gambar notifikasi diset 50% agar pengiriman cepat
                _, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 50])

                # 3. Kirim Notifikasi (Jalan di background thread)
                count = len(current_ids)
                
                # Kirim WhatsApp (via WAHA)
                if CURRENT_CONFIG.get('waha_enabled'):
                    b64 = base64.b64encode(buffer).decode('utf-8')
                    threading.Thread(target=send_whatsapp, args=(self.cam_id, count, b64)).start()
                
                # Kirim Telegram
                if CURRENT_CONFIG.get('telegram_enabled'):
                    img_bytes = buffer.tobytes()
                    threading.Thread(target=send_telegram, args=(self.cam_id, count, img_bytes)).start()

    def get_frame(self):
        """
        Fungsi ini dipanggil oleh Browser/Frontend untuk menampilkan video.
        Sudah dioptimasi agar ringan (Resize & Compress).
        """
        with self.local_lock:
            if self.output_frame is None: return None
            
            # --- OPTIMASI 1: RESIZE (Kecilkan Resolusi Tampilan Web) ---
            height, width = self.output_frame.shape[:2]
            target_width = 640
            
            # Hanya resize jika gambar asli lebih besar dari 640px
            if width > target_width:
                scaling_factor = target_width / float(width)
                display_frame = cv2.resize(self.output_frame, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)
            else:
                display_frame = self.output_frame

            # --- OPTIMASI 2: KOMPRESI JPEG (Turunkan Kualitas) ---
            # Kualitas 50 cukup bagus untuk preview web tapi ukurannya kecil
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 50]
            
            ret, encoded = cv2.imencode(".jpg", display_frame, encode_param)
            return bytearray(encoded) if ret else None
    
    def stop(self):
        self.running = False
        self.join()

# ==========================================
# FUNGSI MANAJEMEN THREAD
# ==========================================
def restart_camera_threads():
    # Matikan stream yang sedang berjalan
    for stream in ACTIVE_STREAMS.values():
        stream.stop()
    ACTIVE_STREAMS.clear()
    
    # Mulai stream baru sesuai config.yaml
    cams = CURRENT_CONFIG.get('cameras', {})
    for cid, src in cams.items():
        if src and str(src).strip():
            stream = CamStream(cid, src)
            stream.daemon = True # Agar thread mati otomatis jika program utama di-close
            stream.start()
            ACTIVE_STREAMS[cid] = stream