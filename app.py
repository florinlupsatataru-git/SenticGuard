import os
import psycopg2
import subprocess
from flask import Flask, request, render_template, redirect, url_for, session, abort
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Load local environment variables from .env
load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "default_secret_key")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD")

def get_db_connection():
    """Connect to PostgreSQL using environment variables"""
    return psycopg2.connect(
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        host=os.getenv("DB_HOST"),
        database=os.getenv("DB_NAME")
    )

def check_rate_limit(ip):
    """Check how many requests an IP has made in the last minute"""
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        one_minute_ago = datetime.now() - timedelta(minutes=1)
        cur.execute("SELECT COUNT(*) FROM security_logs WHERE ip_address = %s AND timestamp > %s", (ip, one_minute_ago))
        count = cur.fetchone()[0]
        cur.close()
        conn.close()
        return count
    except: return 0

def log_event(ip, event_type, severity, desc):
    """Log security events into the PostgreSQL database"""
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        browser = request.headers.get('User-Agent')
        ref = request.headers.get('Referer')
        cur.execute(
            "INSERT INTO security_logs (ip_address, event_type, severity, description, browser, referer) VALUES (%s, %s, %s, %s, %s, %s)",
            (ip, event_type, severity, desc, browser, ref)
        )
        conn.commit()
        cur.close()
        conn.close()
    except Exception as e:
        print(f"Database Logging Error: {str(e)}")

def ban_ip(ip):
    """Add a malicious IP address to the banned_ips database table"""
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("INSERT INTO banned_ips (ip_address) VALUES (%s) ON CONFLICT DO NOTHING", (ip,))
        conn.commit()
        cur.close()
        conn.close()
    except Exception as e:
        print(f"Database Banning Error: {str(e)}")

def is_banned(ip):
    """Check if an IP address exists in the banned_ips table"""
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT 1 FROM banned_ips WHERE ip_address = %s", (ip,))
        banned = cur.fetchone() is not None
        cur.close()
        conn.close()
        return banned
    except: return False

# --- WEBHOOK ROUTE FOR AUTOMATION ---
@app.route('/update_server', methods=['POST'])
def update_server():
    """Endpoint for GitHub Webhook to trigger automatic git pull"""
    try:
        # Extract the real client IP address from behind Nginx proxy for webhook logging
        if request.headers.getlist("X-Forwarded-For"):
            ip = request.headers.getlist("X-Forwarded-For")[0].split(',')[0].strip()
        else:
            ip = request.remote_addr

        # Executes git pull in the project directory
        # Make sure the path matches your server structure
        project_dir = '/home/ubuntu/senticguard'
        subprocess.Popen(['git', 'pull', 'origin', 'main'], cwd=project_dir)
        
        log_event(ip, "AUTO_UPDATE", 1, "Git pull triggered via Webhook")
        return 'Server updated successfully', 200
    except Exception as e:
        return str(e), 500

@app.route('/')
def public_home():
    """Main entry point: logs the visit and redirects to the public Streamlit app"""
    # Extract the real client IP address from behind Nginx proxy
    if request.headers.getlist("X-Forwarded-For"):
        ip = request.headers.getlist("X-Forwarded-For")[0].split(',')[0].strip()
    else:
        ip = request.remote_addr
        
    print(f"[DEBUG] Real visitor IP detected: {ip}")

    if is_banned(ip):
        log_event(ip, "BANNED_ATTEMPT", 10, "Banned IP tried to access gateway")
        abort(403)
    
    log_event(ip, "VISIT", 1, "Redirect to AI Interface via HTTPS")
    
    # Dynamically extract host and redirect to Streamlit app on port 8501
    host_complet = request.host.split(':')[0]
    return redirect(f"https://{host_complet}:8501")

@app.route('/wp-admin')
@app.route('/.env')
@app.route('/config.php')
def honeypot():
    """Honeypot routes to catch and ban malicious scanners"""
    # Extract the real client IP address from behind Nginx proxy
    if request.headers.getlist("X-Forwarded-For"):
        ip = request.headers.getlist("X-Forwarded-For")[0].split(',')[0].strip()
    else:
        ip = request.remote_addr
        
    log_event(ip, "HONEYPOT_HIT", 10, f"Scanner hit forbidden route: {request.path}")
    ban_ip(ip)  # Automatically ban the malicious IP
    abort(403)
