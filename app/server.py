"""Flask 实例与路径常量装配。"""

# 阶段① 原样搬迁自 testV1.0_backup_prompt.py；逻辑零改动。
from flask import Flask, render_template, request, jsonify, send_from_directory
import os
from app.config import MAX_CONTENT_LENGTH, SESSION_FOLDER, STATIC_FOLDER, TEMPLATE_FOLDER, UPLOAD_FOLDER


app = Flask(
    __name__,
    template_folder=TEMPLATE_FOLDER,
    static_folder=STATIC_FOLDER,
    static_url_path="/static",
)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(STATIC_FOLDER, exist_ok=True)
os.makedirs(SESSION_FOLDER, exist_ok=True)
