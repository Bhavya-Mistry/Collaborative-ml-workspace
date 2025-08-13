# generate_keys.py
import bcrypt

password_to_hash = "test123"

# Encode the password to bytes
password_bytes = password_to_hash.encode('utf-8')

# Generate a salt and hash the password
salt = bcrypt.gensalt()
hashed_password_bytes = bcrypt.hashpw(password_bytes, salt)

# Decode the hashed password to a string for the YAML file
hashed_password_str = hashed_password_bytes.decode('utf-8')

print(f"Password: {password_to_hash}")
print(f"Hashed Password (for YAML): {hashed_password_str}")