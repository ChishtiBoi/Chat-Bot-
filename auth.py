from flask import request, session, redirect, url_for, render_template
from flask_bcrypt import Bcrypt
import sqlite3

def init_auth(app):
    bcrypt = Bcrypt(app)

    # Initialize SQLite database for users
    def init_db():
        conn = sqlite3.connect("users.db")
        c = conn.cursor()
        c.execute("""CREATE TABLE IF NOT EXISTS users (
            email TEXT PRIMARY KEY,
            password TEXT,
            name TEXT,
            role TEXT
        )""")
        conn.commit()
        conn.close()

    init_db()

    # Register route
    @app.route("/register", methods=["GET", "POST"])
    def register():
        if request.method == "POST":
            email = request.form["email"]
            password = bcrypt.generate_password_hash(request.form["password"]).decode("utf-8")
            name = request.form["name"]
            role = request.form["role"]
            conn = sqlite3.connect("users.db")
            c = conn.cursor()
            try:
                c.execute("INSERT INTO users (email, password, name, role) VALUES (?, ?, ?, ?)",
                          (email, password, name, role))
                conn.commit()
                return redirect(url_for("login"))
            except sqlite3.IntegrityError:
                conn.close()
                return "Email already registered"
            finally:
                conn.close()
        return render_template("register.html")

    # Login route
    @app.route("/login", methods=["GET", "POST"])
    def login():
        if request.method == "POST":
            email = request.form["email"]
            password = request.form["password"]
            conn = sqlite3.connect("users.db")
            c = conn.cursor()
            c.execute("SELECT * FROM users WHERE email = ?", (email,))
            user = c.fetchone()
            conn.close()
            if user and bcrypt.check_password_hash(user[1], password):
                session["email"] = email
                session["name"] = user[2]
                session["role"] = user[3]
                return redirect(url_for("chat"))
            return "Invalid credentials"
        return render_template("login.html")

    # Logout route
    @app.route("/logout")
    def logout():
        session.clear()
        return redirect(url_for("login"))