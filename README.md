# AI 摄影教学设备

项目已经统一以 `testV1.0_backup_prompt.py` 为业务主程序，`main.py` 是唯一推荐启动入口。旧版 `testV1.0.py` 仅保留作历史参考，不参与当前启动流程。

## 运行

在本目录打开 PowerShell：

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
if (-not (Test-Path _config\api_keys.json)) { New-Item -ItemType Directory -Force -Path _config; Copy-Item api_keys.example.json _config\api_keys.json }
python main.py
```

然后按终端提示打开手机拍照页、电脑显示页或二维码页。上面的复制命令只会在配置文件不存在时创建它，不会覆盖现有密钥。密钥和设备 IP 填在本地 `_config/api_keys.json`；`_config/` 文件夹已被 Git 忽略，不会进入版本库。

## 当前图改图模型

系统按以下顺序测试已配置且可用的模型：

1. `wan2.7-image`
2. `qwen-image-2.0`
3. `doubao-seedream-5.0`（失败时尝试 Lite 或账户 Endpoint）
4. `qwen-image-edit-max`

比较模式默认开启，因此一次“生成”最多会产生 4 次图像模型调用并产生相应费用；没有配置密钥的供应商会被跳过。页面会展示最多 4 张结果，系统先按画幅、清晰度、曝光和明显伪影做技术筛选，用户仍可手动选择最符合意图的一张作为设备主图。

## 优化选项

选项只保留五组互不重复的控制维度，并由后端同一份配置同时驱动页面和接口：

- 优先改善：自动、构图、人物、光线、背景、清晰度
- 目标景别：保持、头像、半身、全身、环境人像
- 画面氛围：保持、自然、暖调、清爽、冷调、电影质感
- 输出画幅：原图、设备 4:3、竖版 3:4、方形、横版 16:9
- 优化强度：保守、标准、明显

每一项都会进入最终图像编辑提示词；输出画幅还会进入各模型的尺寸参数。默认使用 `4:3 设备屏幕`，生成结果可直接适配 320×240 屏幕。其他比例推送到屏幕时采用等比完整显示和留边，不再旋转竖图或强制裁掉人物。

## 自检

以下测试只校验本地逻辑和模拟接口载荷，不会发起付费模型请求：

```powershell
py -3 -m unittest discover -s tests -v
```

提交前还可运行：

```powershell
py -3 -m py_compile main.py testV1.0_backup_prompt.py
git diff --check
git status
```
