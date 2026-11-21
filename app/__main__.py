"""启动入口（横幅/二维码/自动开浏览器/app.run）。"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import os
from app.config import STATIC_FOLDER
from app.contracts import IMAGE_EDIT_MODELS, VISION_MODELS
from app.imagegen import get_ready_models
from app.net import generate_access_qrcodes, get_local_ip, open_browser_later
from app.server import app



if __name__ == "__main__":
    host_ip = get_local_ip()
    qr_info = generate_access_qrcodes(host_ip, port=5000)
    qrcode_url = f"http://127.0.0.1:5000/qrcode"
    display_url = f"http://127.0.0.1:5000/display"

    print("====================================")
    print("服务已准备启动")
    print(f"手机拍照页：{qr_info['mobile_url']}")
    print(f"电脑显示页：{display_url}")
    print(f"二维码页：{qrcode_url}")
    print(f"视觉理解模型顺序：{VISION_MODELS}")
    print(f"图像编辑模型顺序：{get_ready_models(IMAGE_EDIT_MODELS)}")
    print(f"二维码图片已生成：{os.path.join(STATIC_FOLDER, 'qr_mobile.png')}")
    print("====================================")

    # 启动后自动打开二维码页和电脑显示页
    open_browser_later(qrcode_url, delay=1.2)
    open_browser_later(display_url, delay=1.8)

    app.run(host="0.0.0.0", port=5000, debug=True, use_reloader=False)
