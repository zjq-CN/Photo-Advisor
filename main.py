import os
import uuid
from datetime import datetime

from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename

UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "webp"}
MAX_CONTENT_LENGTH = 15 * 1024 * 1024  # 15MB

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH

os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    if "photo" not in request.files:
        return jsonify({"ok": False, "msg": "没有收到图片字段 photo"}), 400

    file = request.files["photo"]
    if file.filename == "":
        return jsonify({"ok": False, "msg": "未选择图片"}), 400

    if not allowed_file(file.filename):
        return jsonify({"ok": False, "msg": "不支持的图片格式，仅支持 jpg/jpeg/png/webp"}), 400

    original_name = secure_filename(file.filename)
    ext = original_name.rsplit(".", 1)[1].lower()
    now_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_id = uuid.uuid4().hex[:8]
    new_filename = f"photo_{now_str}_{unique_id}.{ext}"

    save_path = os.path.join(app.config["UPLOAD_FOLDER"], new_filename)
    file.save(save_path)

    print(f"[已接收] {save_path}")

    return jsonify({
        "ok": True,
        "msg": "图片上传成功，图片已经传回电脑。",
        "filename": new_filename,
        "preview_url": f"/uploads/{new_filename}",
    })


@app.route("/uploads/<path:filename>")
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)


if __name__ == "__main__":
    print("请用电脑真实局域网IP访问，例如：http://192.168.x.x:5000")
    app.run(host="0.0.0.0", port=5000, debug=True)
