import cv2
import time
import threading
import base64
from ultralytics import YOLO
from .globals import CURRENT_CONFIG, ACTIVE_STREAMS, lock
from .plc import trigger_plc
from .notifier import send_whatsapp, send_telegram
import app.core.globals as g

# Load model di global scope
model = YOLO("yolov8n.pt")

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

    def run(self):
        print(f"🎥 Start Cam {self.cam_id} on {self.source}")
        cap = cv2.VideoCapture(self.source)
        
        # Optimize Webcam
        if isinstance(self.source, int):
            cap.set(3, 640)
            cap.set(4, 480)

        while self.running:
            success, frame = cap.read()
            if not success:
                time.sleep(2)
                cap.release()
                cap = cv2.VideoCapture(self.source)
                continue
            
            try:
                conf = float(CURRENT_CONFIG.get('confidence', 0.5))
                # YOLO Process
                results = model.track(frame, persist=True, classes=[0], conf=conf, verbose=False)
                annotated_frame = results[0].plot(line_width=2, font_size=1, conf=False, img=frame)

                if results[0].boxes.id is not None:
                    self.handle_alert(results[0], annotated_frame)
                
                with self.local_lock:
                    self.output_frame = annotated_frame.copy()
            except Exception as e:
                print(f"Error Cam {self.cam_id}: {e}")
                with self.local_lock:
                    self.output_frame = frame

        cap.release()
        print(f"🛑 Cam {self.cam_id} Stopped")

    def handle_alert(self, result, frame):
        now = time.time()
        cooldown = int(CURRENT_CONFIG.get('cooldown', 30))
        
        current_ids = result.boxes.id.cpu().numpy().astype(int)
        new_detection = False
        
        # Cek Global Cooldown (Anti Spam Antar Kamera)
        if (now - g.last_global_send_time > cooldown):
            for pid in current_ids:
                if pid not in self.detected_ids:
                    self.detected_ids.add(pid)
                    new_detection = True
            
            if new_detection:
                g.last_global_send_time = now
                print(f"🔔 Alert Triggered by Cam {self.cam_id}")
                
                # 1. Trigger PLC (Async)
                threading.Thread(target=trigger_plc, args=(self.cam_id,)).start()
                
                # 2. Encode Image
                # _, buffer = cv2.imencode('.jpg', frame)
                _, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 50])

                # 3. Notifiers (Async)
                count = len(current_ids)
                if CURRENT_CONFIG.get('waha_enabled'):
                    b64 = base64.b64encode(buffer).decode('utf-8')
                    threading.Thread(target=send_whatsapp, args=(self.cam_id, count, b64)).start()
                
                if CURRENT_CONFIG.get('telegram_enabled'):
                    img_bytes = buffer.tobytes()
                    threading.Thread(target=send_telegram, args=(self.cam_id, count, img_bytes)).start()

    def get_frame(self):
        with self.local_lock:
            if self.output_frame is None: return None
            ret, encoded = cv2.imencode(".jpg", self.output_frame)
            return bytearray(encoded) if ret else None
    
    def stop(self):
        self.running = False
        self.join()

def restart_camera_threads():
    # Stop existing
    for stream in ACTIVE_STREAMS.values():
        stream.stop()
    ACTIVE_STREAMS.clear()
    
    # Start new based on config
    cams = CURRENT_CONFIG.get('cameras', {})
    for cid, src in cams.items():
        if src and str(src).strip():
            stream = CamStream(cid, src)
            stream.daemon = True
            stream.start()
            ACTIVE_STREAMS[cid] = stream