import os
import yaml
from app.core.globals import CURRENT_CONFIG
from app.core.camera import restart_camera_threads

CONFIG_FILE = 'config.yaml'

def get_defaults():
    return {
        'cameras': {'1': '0', '2': '', '3': '', '4': ''},
        'plc_coils': {'1': 0, '2': 1, '3': 2, '4': 3},
        'modbus_enabled': False, 
        'modbus_type': 'tcp',
        'modbus_ip': '127.0.0.1', 
        'modbus_port': 502,
        'modbus_com': 'COM1', 
        'modbus_baud': 9600, 
        'modbus_slave': 1,
        'confidence': 0.85, 
        'cooldown': 30,
        'waha_enabled': False, 
        'waha_url': 'http://localhost:3001/api/sendImage', 
        'api_key': '',
        'session': 'default', 
        'chat_id': '',
        'telegram_enabled': False, 
        'telegram_token': '', 
        'telegram_chat_id': ''
    }

def load_config():
    defaults = get_defaults()
    if not os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, 'w') as f:
            yaml.dump(defaults, f)
        CURRENT_CONFIG.update(defaults)
    else:
        try:
            with open(CONFIG_FILE, 'r') as f:
                data = yaml.safe_load(f)
                CURRENT_CONFIG.update({**defaults, **(data if data else {})})
                if 'cameras' not in CURRENT_CONFIG: CURRENT_CONFIG['cameras'] = defaults['cameras']
                if 'plc_coils' not in CURRENT_CONFIG: CURRENT_CONFIG['plc_coils'] = defaults['plc_coils']
        except:
            CURRENT_CONFIG.update(defaults)
    return CURRENT_CONFIG

def save_config_from_form(form):
    try:
        data = CURRENT_CONFIG.copy()
        
        # Cameras
        data['cameras'] = {
            '1': form.get('cam_1', ''), 
            '2': form.get('cam_2', ''),
            '3': form.get('cam_3', ''), 
            '4': form.get('cam_4', '')
        }
        
        # PLC
        data['modbus_enabled'] = True if form.get('modbus_enabled') == 'on' else False
        data['modbus_type'] = form.get('modbus_type', 'tcp')
        data['modbus_ip'] = form.get('modbus_ip', '127.0.0.1')
        data['modbus_port'] = int(form.get('modbus_port', 502))
        data['modbus_com'] = form.get('modbus_com', 'COM1')
        data['modbus_baud'] = int(form.get('modbus_baud', 9600))
        data['modbus_slave'] = int(form.get('modbus_slave', 1))
        data['plc_coils'] = {
            '1': int(form.get('coil_1', 0)), 
            '2': int(form.get('coil_2', 1)),
            '3': int(form.get('coil_3', 2)), 
            '4': int(form.get('coil_4', 3))
        }

        # AI & Notif
        data['confidence'] = float(form.get('confidence', 0.85))
        data['cooldown'] = int(form.get('cooldown', 30))
        
        data['waha_enabled'] = True if form.get('waha_enabled') == 'on' else False
        data['waha_url'] = form.get('waha_url', '')
        data['api_key'] = form.get('api_key', '')
        data['session'] = form.get('session', '')
        data['chat_id'] = form.get('chat_id', '')
        
        data['telegram_enabled'] = True if form.get('telegram_enabled') == 'on' else False
        data['telegram_token'] = form.get('telegram_token', '')
        data['telegram_chat_id'] = form.get('telegram_chat_id', '')

        with open(CONFIG_FILE, 'w') as f:
            yaml.dump(data, f)
        
        CURRENT_CONFIG.update(data)
        restart_camera_threads()
        return True
    except Exception as e:
        print(f"Save Config Error: {e}")
        return False