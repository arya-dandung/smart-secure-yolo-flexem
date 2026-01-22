import time
from flask import Blueprint, render_template, request, redirect, url_for, flash, Response
from flask_login import login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from .models import User
from . import db, login_manager
from .utils import load_config, save_config_from_form
from .core.globals import CURRENT_CONFIG, ACTIVE_STREAMS

main_bp = Blueprint('main', __name__)

@login_manager.user_loader
def load_user(uid):
    return db.session.get(User, int(uid))

@main_bp.route('/')
def index():
    return redirect(url_for('main.dashboard' if current_user.is_authenticated else 'main.login'))

@main_bp.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated: return redirect(url_for('main.dashboard'))
    if request.method == 'POST':
        user = User.query.filter_by(username=request.form['username']).first()
        if user and check_password_hash(user.password, request.form['password']):
            login_user(user)
            return redirect(url_for('main.dashboard'))
        flash('Login Gagal', 'danger')
    return render_template('auth/login.html')

@main_bp.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        if User.query.filter_by(username=request.form['username']).first():
            flash('Username sudah dipakai', 'warning')
        else:
            new_user = User(username=request.form['username'], 
                            password=generate_password_hash(request.form['password']))
            db.session.add(new_user)
            db.session.commit()
            return redirect(url_for('main.login'))
    return render_template('auth/register.html')

@main_bp.route('/dashboard', methods=['GET', 'POST'])
@login_required
def dashboard():
    if request.method == 'POST':
        if save_config_from_form(request.form):
            flash('Konfigurasi Disimpan & Sistem Direstart!', 'success')
        else:
            flash('Gagal menyimpan konfigurasi.', 'danger')
        return redirect(url_for('main.dashboard'))
    return render_template('dashboard/index.html', config=CURRENT_CONFIG)

@main_bp.route('/video_feed/<cam_id>')
@login_required
def video_feed(cam_id):
    if cam_id not in ACTIVE_STREAMS: return "Off", 404
    return Response(gen_frames(cam_id), mimetype='multipart/x-mixed-replace; boundary=frame')

def gen_frames(cid):
    stream = ACTIVE_STREAMS.get(cid)
    while True:
        if not stream: break
        frame = stream.get_frame()
        if frame:
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        time.sleep(0.04)

@main_bp.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('main.login'))