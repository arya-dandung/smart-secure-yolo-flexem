# app/__init__.py
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager

db = SQLAlchemy()
login_manager = LoginManager()

def create_app():
    app = Flask(__name__)
    app.config['SECRET_KEY'] = 'rahasia_123'
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

    db.init_app(app)
    login_manager.login_view = 'main.login' # 'main' adalah nama blueprint
    login_manager.init_app(app)

    # 1. Register Blueprint Utama (Dashboard & Login)
    from .routes import main_bp
    app.register_blueprint(main_bp)

    # 2. Register Blueprint SCADA (BARU - TANPA LOGIN)
    # Pastikan Anda sudah membuat file app/scada_routes.py sebelumnya
    from .scada_routes import scada_bp
    app.register_blueprint(scada_bp, url_prefix='/scada')

    return app