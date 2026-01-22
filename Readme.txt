---------------------------------------
file struct
---------------------------------------
smart_security/
│
├── config.yaml              <-- File konfigurasi (otomatis dibuat/diedit)
├── main.py                  <-- File utama untuk menjalankan program (Entry Point)
├── requirements.txt         <-- Daftar library
│
├── app/                     <-- FOLDER APLIKASI UTAMA
│   ├── __init__.py          <-- Setup Flask & Database
│   ├── routes.py            <-- Mengatur URL/Halaman Web (Controller)
│   ├── models.py            <-- Struktur Database User (Model)
│   ├── utils.py             <-- Fungsi bantu (Load/Save Config)
│   │
│   ├── core/                <-- FOLDER LOGIC HARDWARE (OTAK SISTEM)
│   │   ├── __init__.py
│   │   ├── globals.py       <-- Variabel Global (Active Streams, Locks)
│   │   ├── camera.py        <-- Logic Thread Kamera & YOLO
│   │   ├── plc.py           <-- Logic Modbus PLC
│   │   └── notifier.py      <-- Logic WhatsApp & Telegram
│   │
│   └── templates/           <-- FOLDER HTML (VIEW)
│       ├── base.html        <-- Layout Dasar
│       ├── auth/            <-- Folder Login/Register
│       │   ├── login.html
│       │   └── register.html
│       └── dashboard/       <-- Folder Dashboard
│           └── index.html   <-- Dashboard Utama
---------------------------------------------
#video feed   localhost:8000/video_feed/<cam_id>
---------------------------------------------
