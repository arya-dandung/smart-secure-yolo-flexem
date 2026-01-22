import time
from flask import Blueprint, Response, request, abort, current_app, render_template
from .core.globals import ACTIVE_STREAMS

# Buat Blueprint baru bernama 'scada'
scada_bp = Blueprint('scada', __name__)

# --- FUNGSI GENERATOR (TETAP SAMA) ---
def gen_scada_frames(cid):
    stream = ACTIVE_STREAMS.get(cid)
    while stream and stream.running:
        frame = stream.get_frame()
        if frame:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        else:
            time.sleep(0.1)
        time.sleep(0.04)

# --- ROUTE 1: RAW STREAM (SUMBER VIDEO) ---
@scada_bp.route('/stream/<cam_id>')
def stream_feed(cam_id):
    """
    Ini adalah sumber data video mentah.
    Biasanya diakses oleh tag <img> di HTML.
    """
    if cam_id not in ACTIVE_STREAMS:
        return "Camera Offline / Not Found", 404

    return Response(gen_scada_frames(cam_id), 
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# --- ROUTE 2: VIEW HTML (UNTUK SCADA) ---
@scada_bp.route('/view/<cam_id>')
def scada_view(cam_id):
    """
    Gunakan URL ini di Widget 'Web Browser' SCADA.
    Ini akan me-render HTML yang memaksa gambar full screen.
    URL: http://ip:port/scada/view/0
    """
    return render_template('scada_view.html', cam_id=cam_id)