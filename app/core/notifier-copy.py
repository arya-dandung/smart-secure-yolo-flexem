import time
import random
import requests
from .globals import CURRENT_CONFIG

def send_whatsapp(cam_id, count, img_b64):
    try:
        url = CURRENT_CONFIG.get('waha_url')

        # Tambahkan API Key di Header
        headers = {
            "Content-Type": "application/json",
            "X-Api-Key": CURRENT_CONFIG.get('api_key', '') 
        }
        caption = (f"🚨 *ALERT CAM {cam_id}*\n"
                   f"👥 Detect: *{count} People*\n"
                   f"🕒 {time.strftime('%H:%M:%S')}")
        payload = {
            "session": CURRENT_CONFIG.get('session'),
            "chatId": CURRENT_CONFIG.get('chat_id'),
            "file": { 
                "mimetype": "image/jpeg", 
                "filename": "alert.jpg", 
                "data": img_b64 
            },
            "caption": caption
        }
        requests.post(url, json=payload, headers=headers, timeout=10)
        print(f"✅ WhatsApp Sent (Cam {cam_id})")
    except Exception as e:
        print(f"❌ WhatsApp Error: {e}")

def send_telegram(cam_id, count, img_bytes):
    try:
        token = CURRENT_CONFIG.get('telegram_token')
        chat_id = CURRENT_CONFIG.get('telegram_chat_id')
        caption = (f"🚨 <b>ALERT CAM {cam_id}</b>\n"
                   f"👥 Detect: <b>{count} People</b>\n"
                   f"🕒 {time.strftime('%H:%M:%S')}")
        url = f"https://api.telegram.org/bot{token}/sendPhoto"
        files = {'photo': ('alert.jpg', img_bytes, 'image/jpeg')}
        data = {'chat_id': chat_id, 'caption': caption, 'parse_mode': 'HTML'}
        requests.post(url, data=data, files=files, timeout=10)
    except Exception:
        pass