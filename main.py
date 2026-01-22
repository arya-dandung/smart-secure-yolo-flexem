from app import create_app, db
from app.utils import load_config
from app.core.camera import restart_camera_threads

app = create_app()

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    
    # 1. Load Config
    load_config()
    
    # 2. Start Camera Threads
    restart_camera_threads()

    print("🚀 System Online: http://localhost:8000")
    app.run(host='0.0.0.0', port=8000, debug=True, use_reloader=False)