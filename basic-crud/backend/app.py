# app.py
from flask import Flask
from flask_cors import CORS
import os
from dotenv import load_dotenv

from extensions import db, bcrypt

load_dotenv()

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})

# ---- Config ----
app.config['ENV'] = os.getenv('FLASK_ENV')
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DATABASE_URL')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY')
app.config['SQLALCHEMY_ENGINE_OPTIONS'] = {"pool_pre_ping": True}

if not app.config['SQLALCHEMY_DATABASE_URI'] or not app.config['SECRET_KEY']:
    raise RuntimeError("DATABASE_URL and SECRET_KEY must be set in .env")

# ---- Init extensions ----
db.init_app(app)
bcrypt.init_app(app)

# Import AFTER init so models register
from models import User  # noqa
from routes import api_bp

# Mount API
app.register_blueprint(api_bp, url_prefix='/api')

# Dev convenience: create tables if fresh
with app.app_context():
    db.create_all()

if __name__ == '__main__':
    app.run(debug=(app.config['ENV'] == 'development'))
 