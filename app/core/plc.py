import time
import threading
from pymodbus.client import ModbusTcpClient, ModbusSerialClient
from .globals import CURRENT_CONFIG

def _send_modbus_command(coil, state, conf):
    """Helper function to handle a single Modbus transaction (Connect -> Write -> Close)"""
    client = None
    try:
        if conf.get('modbus_type') == 'tcp':
            client = ModbusTcpClient(conf.get('modbus_ip'), port=int(conf.get('modbus_port', 502)))
        else:
            client = ModbusSerialClient(
                port=conf.get('modbus_com'), 
                baudrate=int(conf.get('modbus_baud', 9600)), 
                framer='rtu'
            )

        if client.connect():
            slave = int(conf.get('modbus_slave', 1))
            client.write_coil(coil, state, device_id=slave)
            client.close()
            return True
        else:
            print(f"❌ PLC Connect Failed (State: {state})")
            return False
    except Exception as e:
        print(f"⚠️ PLC Error: {e}")
        if client: client.close()
        return False

def _plc_worker(cam_id, coil, conf):
    """Background worker to handle the pulse timing"""
    # 1. Turn ON
    success = _send_modbus_command(coil, True, conf)
    if success:
        print(f"✅ PLC ON: Cam {cam_id} -> Coil {coil}")
        
        # 2. Wait 5 seconds (Blocking here is fine because we are in a thread)
        time.sleep(5)
        
        # 3. Turn OFF (New Connection)
        _send_modbus_command(coil, False, conf)
        print(f"✅ PLC OFF: Cam {cam_id} -> Coil {coil}")

def trigger_plc(cam_id):
    if not CURRENT_CONFIG.get('modbus_enabled'): return

    conf = CURRENT_CONFIG.copy() # Copy config to prevent changes during thread execution
    coil = int(conf['plc_coils'].get(str(cam_id), 0))

    # Run the logic in a separate thread so it doesn't freeze the camera/app
    t = threading.Thread(target=_plc_worker, args=(cam_id, coil, conf))
    t.daemon = True # Daemon means this thread dies if the main app closes
    t.start()