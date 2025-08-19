# routes.py
from flask import Blueprint, request, jsonify, g
from functools import wraps
import uuid

from extensions import db, bcrypt
from models import User

api_bp = Blueprint('api', __name__)

# ---------- Auth guards ----------

def _get_user_by_api_key():
    api_key = request.headers.get('X-API-Key')
    if not api_key:
        return None
    return User.query.filter_by(api_key=api_key).first()

def auth_required(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        user = _get_user_by_api_key()
        if not user:
            return jsonify({'error': 'Unauthorized: missing or invalid X-API-Key'}), 401
        g.current_user = user
        return f(*args, **kwargs)
    return wrapper

def admin_required(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        user = _get_user_by_api_key()
        if not user or not user.is_admin:
            return jsonify({'error': 'Forbidden: admin access required'}), 403
        g.current_user = user
        return f(*args, **kwargs)
    return wrapper

# ---------- Auth: signup / signin ----------

@api_bp.route('/signup', methods=['POST'])
def signup():
    data = request.get_json(silent=True) or {}
    email = (data.get('email') or '').strip().lower()
    password = data.get('password') or ''

    if not email or not password:
        return jsonify({'error': 'Email and password required'}), 400

    if User.query.filter_by(email=email).first():
        return jsonify({'error': 'Email already exists'}), 409

    hashed_pw = bcrypt.generate_password_hash(password).decode('utf-8')
    api_key = str(uuid.uuid4())

    user = User(email=email, password=hashed_pw, api_key=api_key,
                credits=0, is_admin=False)
    db.session.add(user)
    db.session.commit()

    return jsonify({'message': 'Signup successful', 'api_key': api_key}), 201


@api_bp.route('/signin', methods=['POST'])
def signin():
    data = request.get_json(silent=True) or {}
    email = (data.get('email') or '').strip().lower()
    password = data.get('password') or ''

    if not email or not password:
        return jsonify({'error': 'Email and password required'}), 400

    user = User.query.filter_by(email=email).first()
    if not user or not bcrypt.check_password_hash(user.password, password):
        return jsonify({'error': 'Invalid credentials'}), 401

    return jsonify({'message': 'Signin successful', 'api_key': user.api_key}), 200

# ---------- User (self) ----------

@api_bp.route('/me', methods=['GET'])
@auth_required
def me():
    return jsonify(g.current_user.as_dict()), 200

@api_bp.route('/me', methods=['PATCH'])
@auth_required
def update_me():
    """
    Users can add/update their details (profile fields + optionally email & password).
    """
    u = g.current_user
    data = request.get_json(silent=True) or {}

    # Optional email change (ensure unique)
    if 'email' in data:
        new_email = (data['email'] or '').strip().lower()
        if not new_email:
            return jsonify({'error': 'Email cannot be empty'}), 400
        if new_email != u.email and User.query.filter_by(email=new_email).first():
            return jsonify({'error': 'Email already exists'}), 409
        u.email = new_email

    # Optional password change
    if 'password' in data and data['password']:
        u.password = bcrypt.generate_password_hash(data['password']).decode('utf-8')

    # Profile fields
    for fld in ['first_name', 'last_name', 'phone', 'address', 'city', 'country']:
        if fld in data:
            setattr(u, fld, data.get(fld))

    db.session.commit()
    return jsonify({'message': 'Profile updated', 'me': u.as_dict()}), 200

@api_bp.route('/me', methods=['DELETE'])
@auth_required
def delete_me():
    """
    Users can delete their account.
    """
    u = g.current_user
    db.session.delete(u)
    db.session.commit()
    return jsonify({'message': 'Account deleted'}), 200

@api_bp.route('/me/dashboard', methods=['GET'])
@auth_required
def my_dashboard():
    """
    Returns table-friendly rows to render immediately.
    """
    u = g.current_user
    return jsonify({
        "title": "My Dashboard",
        "rows": u.as_table_rows()
    }), 200

# ---------- Admin: CRUD on users ----------

@api_bp.route('/admin/users', methods=['GET'])
@admin_required
def admin_list_users():
    """
    Supports pagination & search:
      - ?page=1&per_page=20
      - ?q=alice (search in email, first_name, last_name, phone)
    """
    page = max(int(request.args.get('page', 1)), 1)
    per_page = min(max(int(request.args.get('per_page', 20)), 1), 100)
    q = (request.args.get('q') or '').strip().lower()

    query = User.query
    if q:
        like = f"%{q}%"
        query = query.filter(
            db.or_(
                User.email.ilike(like),
                User.first_name.ilike(like),
                User.last_name.ilike(like),
                User.phone.ilike(like),
            )
        )

    pagination = query.order_by(User.id.asc()).paginate(page=page, per_page=per_page, error_out=False)
    users = [u.as_dict() for u in pagination.items]
    return jsonify({
        "items": users,
        "page": page,
        "per_page": per_page,
        "total": pagination.total,
        "pages": pagination.pages
    }), 200

@api_bp.route('/admin/users', methods=['POST'])
@admin_required
def admin_create_user():
    data = request.get_json(silent=True) or {}
    email = (data.get('email') or '').strip().lower()
    password = data.get('password') or ''
    if not email or not password:
        return jsonify({'error': 'Email and password required'}), 400
    if User.query.filter_by(email=email).first():
        return jsonify({'error': 'Email already exists'}), 409

    hashed_pw = bcrypt.generate_password_hash(password).decode('utf-8')
    api_key = str(uuid.uuid4())

    u = User(
        email=email,
        password=hashed_pw,
        api_key=api_key,
        credits=int(data.get('credits', 0) or 0),
        is_admin=bool(data.get('is_admin', False)),
        first_name=data.get('first_name'),
        last_name=data.get('last_name'),
        phone=data.get('phone'),
        address=data.get('address'),
        city=data.get('city'),
        country=data.get('country'),
    )
    db.session.add(u)
    db.session.commit()
    return jsonify(u.as_dict()), 201

@api_bp.route('/admin/users/<int:user_id>', methods=['GET'])
@admin_required
def admin_get_user(user_id):
    u = User.query.get_or_404(user_id)
    return jsonify(u.as_dict()), 200

@api_bp.route('/admin/users/<int:user_id>', methods=['PATCH'])
@admin_required
def admin_update_user(user_id):
    u = User.query.get_or_404(user_id)
    data = request.get_json(silent=True) or {}
    self_edit = (u.id == g.current_user.id)

    # ... (same email/password/profile/credits code as before)

    if 'is_admin' in data:
        new_is_admin = bool(data['is_admin'])
        if u.is_admin and not new_is_admin:
            admins_count = User.query.filter_by(is_admin=True).count()
            if admins_count <= 1:
                return jsonify({'error': 'Cannot remove admin rights from the last remaining admin'}), 400
        u.is_admin = new_is_admin

    db.session.commit()

    resp = {'message': 'User updated', 'user': u.as_dict()}
    if self_edit and not u.is_admin:
        resp['self_demoted'] = True
    return jsonify(resp), 200


@api_bp.route('/admin/users/<int:user_id>', methods=['DELETE'])
@admin_required
def admin_delete_user(user_id):
    u = User.query.get_or_404(user_id)

    # 1) prevent deleting yourself from the admin panel
    if u.id == g.current_user.id:
        return jsonify({
            'error': 'Admins cannot delete themselves via the admin panel. Use /api/me (Profile → Delete account) or sign in as another admin.'
        }), 400

    # 2) optional: prevent deleting the last remaining admin
    if u.is_admin:
        admins_count = User.query.filter_by(is_admin=True).count()
        if admins_count <= 1:
            return jsonify({'error': 'Cannot delete the last remaining admin'}), 400

    db.session.delete(u)
    db.session.commit()
    return jsonify({'message': 'User deleted'}), 200
