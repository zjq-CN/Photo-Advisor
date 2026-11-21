"""本机 IP/二维码/浏览器启动。"""

# 阶段① 原样搬迁自 testV1.0_backup_prompt.py；本文件逻辑零改动（D10 例外见迁移表）。
from typing import Any, Callable, Dict, List, Optional, Tuple
import os
import qrcode
import re
import subprocess
import threading
import time
import webbrowser
from app.config import STATIC_FOLDER



def open_browser_later(url: str, delay: float = 1.2) -> None:
    """
    Flask 启动后稍等片刻，再自动打开默认浏览器
    """
    def _open():
        time.sleep(delay)
        try:
            webbrowser.open(url)
            print(f"[自动打开] {url}")
        except Exception as e:
            print(f"[自动打开失败] {e}")

    threading.Thread(target=_open, daemon=True).start()


# =========================
# Windows ipconfig 提取局域网IP
# =========================
def get_local_ip() -> str:
    """
    从中英文 Windows ipconfig 输出中提取局域网 IPv4。
    优先无线网卡，其次物理以太网，最后才使用其他非虚拟适配器。
    """
    try:
        result = subprocess.run(
            ["ipconfig"],
            capture_output=True,
            text=True,
            encoding="gbk",
            errors="ignore"
        )
        text = result.stdout.replace("\r\n", "\n")
        # 直接在中英文 adapter 标题前切分；不能按空行切，因为部分 Windows
        # 版本会在网卡标题和字段之间也插入空行。
        blocks = re.split(
            r"(?im)(?=^[^\n]*(?:adapter|适配器)[^:\n]*:\s*$)",
            text,
        )

        ethernet_ip: Optional[str] = None
        fallback_ip: Optional[str] = None
        virtual_markers = (
            "virtual", "vethernet", "vmware", "hyper-v", "bluetooth",
            "loopback", "tunnel", "虚拟", "隧道", "蓝牙",
        )

        for block in blocks:
            block_strip = block.strip()
            if not block_strip:
                continue

            title = block_strip.split("\n")[0].lower()

            m = re.search(r"IPv4[^:\n]*:\s*([0-9]+\.[0-9]+\.[0-9]+\.[0-9]+)", block, re.IGNORECASE)
            if not m:
                continue

            ip = m.group(1)
            if ip.startswith("169.254.") or ip == "127.0.0.1":
                continue

            if (
                "无线局域网适配器" in title
                or "wireless lan" in title
                or "wlan" in title
                or "wi-fi" in title
                or "wifi" in title
            ):
                print(f"[IP识别] 选择无线网卡: {title}")
                print(f"[IP识别] IPv4: {ip}")
                return ip

            is_virtual = any(marker in title for marker in virtual_markers)
            if not is_virtual and ("以太网适配器" in title or "ethernet adapter" in title):
                ethernet_ip = ethernet_ip or ip
            elif not is_virtual:
                fallback_ip = fallback_ip or ip

        if ethernet_ip:
            print("[IP识别] 未找到无线网卡，改用以太网 IPv4")
            print(f"[IP识别] IPv4: {ethernet_ip}")
            return ethernet_ip

        if fallback_ip:
            print("[IP识别] 使用其他活动网卡 IPv4")
            print(f"[IP识别] IPv4: {fallback_ip}")
            return fallback_ip

    except Exception as e:
        print(f"[IP识别] 解析 ipconfig 失败: {e}")

    return "127.0.0.1"


def generate_qr_image(text: str, save_path: str) -> None:
    qr = qrcode.QRCode(
        version=1,
        box_size=10,
        border=2,
    )
    qr.add_data(text)
    qr.make(fit=True)
    img = qr.make_image(fill_color="black", back_color="white")
    img.save(save_path)


def generate_access_qrcodes(host_ip: str, port: int = 5000) -> Dict[str, str]:
    mobile_url = f"http://{host_ip}:{port}/mobile"

    mobile_qr_path = os.path.join(STATIC_FOLDER, "qr_mobile.png")
    generate_qr_image(mobile_url, mobile_qr_path)

    return {
        "mobile_url": mobile_url,
        "mobile_qr": "/static/qr_mobile.png",
    }
