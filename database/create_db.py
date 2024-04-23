import sqlite3
from passlib.hash import pbkdf2_sha256
import logging
# Create a connection to the SQLite database
conn_user = sqlite3.connect("user_db.db", check_same_thread=False)
cursor = conn_user.cursor()

cursor.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY,
        username TEXT,
        email TEXT,
        password TEXT
    )
""")


conn_user.commit()


# def register_user(username, email, password):
#     encrypted_password = pbkdf2_sha256.hash(password)
#     try:
#         cursor.execute("INSERT INTO users (username, email, password) VALUES (?, ?, ?)", (username, email, encrypted_password))
#         conn_user.commit()
#         return True
#     except sqlite3.IntegrityError:
#         return False

def register_user(username, email, password):
    encrypted_password = pbkdf2_sha256.hash(password)
    try:
        with sqlite3.connect("user_db.db", check_same_thread=False) as conn_user:
            cursor = conn_user.cursor()
            cursor.execute("INSERT INTO users (username, email, password) VALUES (?, ?, ?)",
                           (username, email, encrypted_password))
            conn_user.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    
def login_user(username, password):
    cursor.execute("SELECT * FROM users WHERE username = ?", (username,))
    user = cursor.fetchone()
    if user and pbkdf2_sha256.verify(password, user[3]):
        return True
    return False

# def login_user(username, password):
#     try:
#         with sqlite3.connect("user_db.db", check_same_thread=False) as conn_user:
#             cursor = conn_user.cursor()
#             cursor.execute("SELECT * FROM users WHERE username = ?", (username,))
#             user = cursor.fetchone()
#             if user and pbkdf2_sha256.verify(password, user[3]):
#                 return True
#     except sqlite3.Error:
#         return False
#     return False

def user_exists(username):
    try:
        with sqlite3.connect("user_db.db", check_same_thread=False) as conn_user:
            cursor = conn_user.cursor()
            cursor.execute("SELECT * FROM users WHERE username = ?", (username,))
            user = cursor.fetchone()
            if user:
                return True
    except sqlite3.Error:
        return False
    return False

def email_exists(email):
    try:
        with sqlite3.connect("user_db.db", check_same_thread=False) as conn_user:
            cursor = conn_user.cursor()
            cursor.execute("SELECT * FROM users WHERE email = ?", (email,))
            user = cursor.fetchone()
            if user:
                return True
    except sqlite3.Error:
        return False
    return False
    
# def login_user(username, password):
#     cursor.execute("SELECT * FROM users WHERE username = ?", (username,))
#     user = cursor.fetchone()
#     if user and pbkdf2_sha256.verify(password, user[3]):
#         return True
#     return False

# def user_exists(username):
#     cursor.execute("SELECT * FROM users WHERE username = ?", (username,))
#     user = cursor.fetchone()
#     if user:
#         return True
#     return False

# def email_exists(email):
#     cursor.execute("SELECT * FROM users WHERE email = ?", (email,))
#     user = cursor.fetchone()
#     if user:
#         return True
#     return False
