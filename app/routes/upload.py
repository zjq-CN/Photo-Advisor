"""上传路由（转调 api_diagnose）。"""

# 阶段① 原样搬迁自 testV1.0_backup_prompt.py；本文件逻辑零改动（D10 例外见迁移表）。
from app.routes.api import api_diagnose
from app.server import app



@app.route("/upload", methods=["POST"])
def upload():
    return api_diagnose()
