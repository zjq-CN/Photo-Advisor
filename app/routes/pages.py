"""页面路由。"""

# 阶段① 原样搬迁自 testV1.0_backup_prompt.py；本文件逻辑零改动（D10 例外见迁移表）。
from flask import Flask, render_template, request, jsonify, send_from_directory
from app.contracts import DEFAULT_USER_INTENT, OPTIMIZATION_SCHEMA
from app.net import generate_access_qrcodes, get_local_ip
from app.server import app



# =========================
# 路由
# =========================
@app.route("/")
def root():
    return render_template(
        "mobile.html",
        optimization_schema=OPTIMIZATION_SCHEMA,
        default_user_intent=DEFAULT_USER_INTENT,
    )


@app.route("/mobile")
def mobile_page():
    return render_template(
        "mobile.html",
        optimization_schema=OPTIMIZATION_SCHEMA,
        default_user_intent=DEFAULT_USER_INTENT,
    )


@app.route("/display")
def display_page():
    return render_template("display.html")


@app.route("/qrcode")
def qrcode_page():
    host_ip = get_local_ip()
    qr_info = generate_access_qrcodes(host_ip, port=5000)
    return render_template("qrcode.html", qr_info=qr_info)


@app.route("/uploads/<path:filename>")
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)
