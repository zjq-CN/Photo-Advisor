"""P.A 应用包。装配：server 建 Flask 实例，routes 注册路由。"""

from app.server import app  # noqa: F401

from app.routes import api, pages, upload  # noqa: F401,E402  # 触发路由注册
