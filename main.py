"""项目统一启动入口。

实际业务主程序为 app/__main__.py（阶段①迁移后）。回滚开关 = 下面 PRIMARY_PROGRAM 一行。
使用 `python main.py` 即可启动，避免再误运行旧版 testV1.0.py。
"""

from pathlib import Path
import runpy


PRIMARY_PROGRAM = Path(__file__).parent / "app" / "__main__.py"


def main() -> None:
    if not PRIMARY_PROGRAM.is_file():
        raise FileNotFoundError(f"主程序不存在：{PRIMARY_PROGRAM}")
    runpy.run_path(str(PRIMARY_PROGRAM), run_name="__main__")


if __name__ == "__main__":
    main()
