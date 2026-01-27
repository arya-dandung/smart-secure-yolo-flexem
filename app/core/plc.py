# import time
# import threading
# from pymodbus.client import ModbusTcpClient, ModbusSerialClient
# from .globals import CURRENT_CONFIG

# def _send_modbus_command(coil, state, conf):
#     """Helper function to handle a single Modbus transaction (Connect -> Write -> Close)"""
#     client = None
#     try:
#         if conf.get('modbus_type') == 'tcp':
#             client = ModbusTcpClient(conf.get('modbus_ip'), port=int(conf.get('modbus_port', 502)))
#         else:
#             client = ModbusSerialClient(
#                 port=conf.get('modbus_com'), 
#                 baudrate=int(conf.get('modbus_baud', 9600)), 
#                 framer='rtu'
#             )

#         if client.connect():
#             slave = int(conf.get('modbus_slave', 1))
#             client.write_coil(coil, state, device_id=slave)
#             client.close()
#             return True
#         else:
#             print(f"❌ PLC Connect Failed (State: {state})")
#             return False
#     except Exception as e:
#         print(f"⚠️ PLC Error: {e}")
#         if client: client.close()
#         return False

# def _plc_worker(cam_id, coil, conf):
#     """Background worker to handle the pulse timing"""
#     # 1. Turn ON
#     success = _send_modbus_command(coil, True, conf)
#     if success:
#         print(f"✅ PLC ON: Cam {cam_id} -> Coil {coil}")
        
#         # 2. Wait 5 seconds (Blocking here is fine because we are in a thread)
#         time.sleep(5)
        
#         # 3. Turn OFF (New Connection)
#         _send_modbus_command(coil, False, conf)
#         print(f"✅ PLC OFF: Cam {cam_id} -> Coil {coil}")

# def trigger_plc(cam_id):
#     if not CURRENT_CONFIG.get('modbus_enabled'): return

#     conf = CURRENT_CONFIG.copy() # Copy config to prevent changes during thread execution
#     coil = int(conf['plc_coils'].get(str(cam_id), 0))

#     # Run the logic in a separate thread so it doesn't freeze the camera/app
#     t = threading.Thread(target=_plc_worker, args=(cam_id, coil, conf))
#     t.daemon = True # Daemon means this thread dies if the main app closes
#     t.start()

import time
import threading
from pymodbus.client import ModbusTcpClient, ModbusSerialClient
from .globals import CURRENT_CONFIG

# ==========================================
# GLOBAL STATE VARIABLES
# ==========================================
# Menyimpan status terakhir Coil (True=ON, False=OFF) agar tidak spam perintah ke PLC
PLC_STATES = {} 

# Menyimpan waktu terakhir kali orang terdeteksi (Unix Timestamp)
LAST_SEEN = {}

# Konfigurasi Delay (Hysteresis)
# Lampu akan tetap menyala selama 3 detik setelah orang menghilang
# Ini mencegah lampu kedip-kedip jika tracking YOLO lepas sesaat
OFF_DELAY_SECONDS = 3.0 

# ==========================================
# MODBUS HELPER (Low Level)
# ==========================================
def _send_modbus_command(coil, state, conf):
    """
    Fungsi dasar untuk melakukan koneksi dan write coil ke PLC.
    Mendukung TCP dan Serial (RTU).
    """
    client = None
    try:
        # Cek tipe koneksi dari config
        if conf.get('modbus_type') == 'tcp':
            client = ModbusTcpClient(
                conf.get('modbus_ip'), 
                port=int(conf.get('modbus_port', 502))
            )
        else:
            client = ModbusSerialClient(
                port=conf.get('modbus_com'), 
                baudrate=int(conf.get('modbus_baud', 9600)), 
                framer='rtu',
                timeout=1
            )

        if client.connect():
            slave_id = int(conf.get('modbus_slave', 1))
            # Write single coil
            client.write_coil(coil, state, device_id=slave_id)
            client.close()
            return True
        else:
            print(f"❌ PLC Connect Failed (Target: Coil {coil} -> {state})")
            return False
            
    except Exception as e:
        print(f"⚠️ PLC Communication Error: {e}")
        if client: 
            client.close()
        return False

def _set_coil_async(cam_id, coil, state, conf):
    """
    Worker thread untuk mengirim perintah agar tidak memblokir kamera.
    Jika gagal, kita reset status memori agar sistem mencoba lagi nanti.
    """
    success = _send_modbus_command(coil, state, conf)
    
    if success:
        status_str = "ON 🚨" if state else "OFF ✅"
        print(f"[PLC] Cam {cam_id} -> Coil {coil} set to {status_str}")
    else:
        # PENTING: Jika gagal kirim ke alat, kembalikan status memori ke posisi lawan
        # Supaya di frame berikutnya sistem mencoba mengirim ulang.
        global PLC_STATES
        PLC_STATES[cam_id] = not state 

# ==========================================
# MAIN LOGIC (Dipanggil dari Camera)
# ==========================================
def update_plc_status(cam_id, is_person_detected):
    """
    Fungsi ini dipanggil SETIAP FRAME oleh camera.py.
    Mengatur logika kapan harus ON dan kapan harus OFF.
    """
    # 1. Cek apakah fitur Modbus diaktifkan
    if not CURRENT_CONFIG.get('modbus_enabled'): 
        return

    # 2. Persiapkan Config & Coil ID
    conf = CURRENT_CONFIG.copy()
    coil_str = conf['plc_coils'].get(str(cam_id))
    
    if coil_str is None: 
        return # Skip jika kamera ini tidak punya mapping coil
        
    coil = int(coil_str)
    current_time = time.time()
    
    # 3. Ambil status saat ini dari memori (Default False/Mati)
    is_currently_on = PLC_STATES.get(cam_id, False)
    
    # === LOGIKA DETEKSI ===
    if is_person_detected:
        # Update waktu terakhir melihat orang
        LAST_SEEN[cam_id] = current_time
        
        # JIKA SEBELUMNYA MATI -> NYALAKAN SEKARANG
        if not is_currently_on:
            PLC_STATES[cam_id] = True # Kunci status di memori dulu
            threading.Thread(target=_set_coil_async, args=(cam_id, coil, True, conf)).start()
            
    # === LOGIKA TIDAK ADA DETEKSI (CLEAR) ===
    else:
        last_time = LAST_SEEN.get(cam_id, 0)
        
        # Hitung durasi sejak orang menghilang
        elapsed = current_time - last_time
        
        # JIKA SEBELUMNYA NYALA dan SUDAH LEWAT DURASI DELAY (3 detik) -> MATIKAN
        if is_currently_on and elapsed > OFF_DELAY_SECONDS:
            PLC_STATES[cam_id] = False # Kunci status
            threading.Thread(target=_set_coil_async, args=(cam_id, coil, False, conf)).start()