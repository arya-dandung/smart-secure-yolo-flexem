# Smart Security System 🛡️ (YOLOv8 + Flexem PLC Integration)

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![YOLOv8](https://img.shields.io/badge/AI-YOLOv8-green)
![Flask](https://img.shields.io/badge/Web-Flask-red)
![Status](https://img.shields.io/badge/Status-Active-success)

Sistem keamanan cerdas berbasis Computer Vision yang mengintegrasikan deteksi objek real-time (YOLOv8) dengan sistem otomasi industri (**PLC Flexem**) dan notifikasi instan.

Project ini dirancang untuk mendeteksi intrusi manusia, memicu alarm fisik melalui PLC, dan mengirimkan bukti gambar ke WhatsApp & Telegram secara real-time dengan manajemen resource yang efisien.

## 🌟 Fitur Utama

- **AI Powered Detection**: Menggunakan YOLOv8n (Nano) untuk deteksi manusia yang cepat dan akurat.
- **Multi-Camera Threading**: Mendukung banyak kamera (Webcam/RTSP/CCTV) secara bersamaan tanpa blocking.
- **Industrial Integration**: Terintegrasi dengan **PLC Flexem** (via Modbus/TCP) untuk memicu sirine/lampu strobo saat terdeteksi bahaya.
- **Smart Notifications**:
  - **WhatsApp**: Kirim alert + foto via WAHA API.
  - **Telegram**: Kirim alert + foto via Telegram Bot.
  - **Auto Compression**: Gambar dikompresi (Quality 70%) untuk pengiriman cepat dan hemat bandwidth.
- **Web Dashboard**: Antarmuka berbasis Flask untuk memantau stream video dan mengubah konfigurasi tanpa coding.
- **Smart Cooldown**: Mencegah spam notifikasi dengan sistem *global cooldown* dan *ID tracking*.

## 🛠️ Tech Stack

- **Core**: Python 3.x
- **Computer Vision**: OpenCV, Ultralytics YOLOv8
- **Web Framework**: Flask, Flask-Login, Flask-SQLAlchemy
- **Industrial Protocol**: PyModbus (untuk komunikasi ke PLC)
- **Database**: SQLite (untuk manajemen user & log)

## ⚙️ Instalasi

1. **Clone Repository**
   ```bash
   git clone [https://github.com/arya-dandung/smart-secure-yolo-flexem.git](https://github.com/arya-dandung/smart-secure-yolo-flexem.git)
   cd smart-secure-yolo-flexem
Buat Virtual Environment (Disarankan)

Bash

python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
Install Dependencies

Bash

pip install -r requirements.txt
Konfigurasi Database Saat pertama kali dijalankan, sistem akan otomatis membuat file instance/database.db.

🚀 Cara Menjalankan
Jalankan perintah berikut untuk memulai server:

Bash

python main.py
Akses Dashboard melalui browser di: http://localhost:5000

Default Username: (Buat via halaman Register)

Default Password: (Buat via halaman Register)

📝 Konfigurasi Sistem
Konfigurasi dapat diatur melalui file config.yaml atau langsung melalui menu Settings di Dashboard:

Cameras: Tambah URL RTSP atau Index Webcam (0, 1).

PLC: Atur IP Address PLC Flexem dan Register Address.

Confidence: Ambang batas akurasi AI (0.1 - 1.0).

Notifikasi: Token API WhatsApp dan Bot Token Telegram.

📂 Struktur Project
smart-secure-yolo-flexem/
├── app/
│   ├── core/           # Logika Utama (Camera, PLC, Notifier)
│   ├── templates/      # Tampilan Web (HTML)
│   ├── routes.py       # Endpoint Flask
│   └── models.py       # Database Model
├── config.yaml         # File Konfigurasi
├── main.py             # Entry Point
└── requirements.txt    # Daftar Library

C:\AI SATPAM\smart-secure-yolo-flexem-main\  <-- Root Project
│
├── known_faces/              <-- [BARU] Folder tempat taruh foto wajah (arya.jpg, bos.jpg)
│   ├── arya.jpg
│   └── ...
│
├── app/
│   ├── core/
│   │   ├── __init__.py
│   │   ├── camera.py         <-- [EDIT] Kode utama CCTV ada di sini
│   │   ├── face_engine.py    <-- [BARU] Logika InsightFace (CPU)
│   │   ├── globals.py
│   │   ├── notifier.py
│   │   └── plc.py
│   │
│   └── ... (folder lain)
│
├── main.py                   <-- Entry point aplikasi
└── requirements.txt
