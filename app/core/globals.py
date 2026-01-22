# app/core/globals.py
import threading

# Menyimpan konfigurasi saat ini di memori
CURRENT_CONFIG = {}

# Menyimpan objek thread kamera yang aktif
ACTIVE_STREAMS = {}

# Lock untuk thread safety
lock = threading.Lock()

# Global timer agar notifikasi tidak spam
last_global_send_time = 0