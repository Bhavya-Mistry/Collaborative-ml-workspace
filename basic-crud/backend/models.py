# models.py
from extensions import db, bcrypt

class User(db.Model):
    __tablename__ = 'usertable'

    id = db.Column(db.Integer, primary_key=True)

    # Auth / identity
    email = db.Column(db.String(120), unique=True, nullable=False, index=True)
    password = db.Column(db.String(128), nullable=False)            # bcrypt hash
    api_key = db.Column(db.String(128), unique=True, nullable=False)

    # Profile "details"
    first_name = db.Column(db.String(80))
    last_name  = db.Column(db.String(80))
    phone      = db.Column(db.String(20))
    address    = db.Column(db.String(255))
    city       = db.Column(db.String(80))
    country    = db.Column(db.String(80))

    # System flags / misc
    credits  = db.Column(db.Integer, nullable=False, default=0)
    is_admin = db.Column(db.Boolean, nullable=False, default=False)

    # Timestamps
    created_at = db.Column(db.DateTime, server_default=db.func.now(), nullable=False)
    updated_at = db.Column(db.DateTime, server_default=db.func.now(),
                           onupdate=db.func.now(), nullable=False)

    # Helpers
    def set_password(self, plain_password: str):
        self.password = bcrypt.generate_password_hash(plain_password).decode('utf-8')

    def check_password(self, plain_password: str) -> bool:
        return bcrypt.check_password_hash(self.password, plain_password)

    def as_table_rows(self):
        """
        Returns a list of {key, value} rows — easy to render as a table on the frontend.
        """
        return [
            {"key": "ID", "value": self.id},
            {"key": "Email", "value": self.email},
            {"key": "First name", "value": self.first_name or ""},
            {"key": "Last name", "value": self.last_name or ""},
            {"key": "Phone", "value": self.phone or ""},
            {"key": "Address", "value": self.address or ""},
            {"key": "City", "value": self.city or ""},
            {"key": "Country", "value": self.country or ""},
            {"key": "Credits", "value": self.credits},
            {"key": "Admin", "value": bool(self.is_admin)},
            {"key": "Created at", "value": self.created_at.isoformat() if self.created_at else None},
            {"key": "Updated at", "value": self.updated_at.isoformat() if self.updated_at else None},
        ]

    def as_dict(self):
        return {
            "id": self.id,
            "email": self.email,
            "first_name": self.first_name,
            "last_name": self.last_name,
            "phone": self.phone,
            "address": self.address,
            "city": self.city,
            "country": self.country,
            "credits": self.credits,
            "is_admin": self.is_admin,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }
