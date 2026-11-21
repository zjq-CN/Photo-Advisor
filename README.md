# AI 摄影教学设备（Photo Advisor）

一台 Windows 电脑 + 一块 ESP32-S3（320×240）小屏，构成一套面向中学摄影课的**课堂即时反馈装置**：

学生用手机拍一张照片，20 秒内看到"这张照片最该改的是哪一处、下次怎么拍"，并在屏幕上看到 AI 按教学意图改出来的理想效果图。

它不教参数。它只回答学生拍完之后最想问的那一个问题。

---

## 它解决什么问题

摄影课上最贵的一环是**从拍完到知道自己错在哪**的这段等待。老师一次要顾几十个人，学生往往到下课前也不知道自己那张照片差在哪。

这个项目把这段反馈压到 20 秒内，并且做到三件事：

- **说人话**：不给"构图失衡""光比过大"这种词，给"人物偏右，右边空得太多，往左站半步"。
- **只给一条**：每次只指出最要紧的一处，不列十条让学生无从下手。
- **看得见**：改完的效果图直接投到小屏和电脑屏上，学生能对比"改前 / 改后"。

---

## 一次完整的课堂使用

1. 老师在电脑上启动程序，浏览器自动打开**二维码页**和**显示页**；
2. 学生扫码进入**手机拍照页**，拍一张，选择"标准诊断"或"快速诊断"；
3. 后端调用视觉模型做 10 项专业诊断，手机端用一句话给出结论 + 一条可执行的调整建议；
4. 学生按兴趣选择 5 组优化方向（优先改善 / 目标景别 / 画面氛围 / 输出画幅 / 优化强度）；
5. 后端把诊断 + 意图组装成编辑提示词，并发调用多个国产图像编辑模型生成候选图；
6. 候选图自动推到 ESP32 小屏，电脑显示页同步展示诊断报告与多模型对比；
7. 老师可以点任意一张候选图重新推屏，用于课堂讲解。

---

## 目录结构

```
P.A/
├── main.py                 # 唯一启动入口：python main.py（转发到 app/__main__.py）
├── app/                    # 业务代码（按职责拆分，共 24 个模块）
│   ├── __main__.py         # 启动横幅 + 二维码生成 + 打开浏览器 + app.run
│   ├── server.py           # Flask 实例
│   ├── config.py           # 配置读取（密钥、ESP 参数），带 mtime 缓存
│   ├── contracts.py        # OPTIMIZATION_SCHEMA / DEFAULT_USER_INTENT / 意图归一化
│   ├── vision.py           # 视觉诊断：模型链、双模式、探测与降级
│   ├── parsing.py          # 模型返回的 JSON 容错解析
│   ├── prompts.py          # 诊断提示词与重建提示词模板
│   ├── generation.py       # 生成编排：多模型比较、候选回调、增量写回
│   ├── strategy.py         # 策略生成与编辑提示词组装（含 9 条硬约束）
│   ├── esp.py              # 推送队列、图片压缩、失败即停
│   ├── net.py              # 局域网地址识别、二维码生成
│   ├── quality.py          # OpenCV 画幅/清晰度/曝光/伪影评分
│   ├── storage.py          # 会话文件与 latest_result.json 读写
│   ├── imagegen/           # 各供应商 runner：dashscope / doubao / hunyuan / jimeng / zhipu
│   └── routes/             # pages.py / api.py / upload.py
├── geometry.py             # 可选：本地 MediaPipe 人脸几何参考（失败即跳过，不影响主流程）
├── templates/              # mobile.html / display.html / qrcode.html（原生 HTML/CSS/JS，无构建）
├── static/                 # latest_result.json、sessions/、二维码图片
├── uploads/                # 上传原图与生成图（已 gitignore）
├── tests/                  # 单元测试 + 重构基线快照（不发付费请求）
├── legacy/                 # 重构前的单文件版本，冻结留档，不参与运行
├── tools/push_to_github.ps1 # 直连 github.com 不通时的推送脚本（见下文）
├── requirements.txt
├── api_keys.example.json   # 配置模板
└── CHANGELOG.md
```

---

## 快速开始

需要 Python 3.10 / 3.11。在本目录打开 PowerShell：

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt

if (-not (Test-Path _config\api_keys.json)) {
    New-Item -ItemType Directory -Force -Path _config
    Copy-Item api_keys.example.json _config\api_keys.json
}

python main.py
```

启动后终端会打印手机拍照页、电脑显示页、二维码页三个地址，并自动打开后两个。手机与电脑必须在**同一局域网**。

复制命令只在配置文件不存在时创建它，不会覆盖已有密钥。`_config/` 已被 Git 忽略。

---

## 配置

密钥与设备参数都在 `_config/api_keys.json`（从 `api_keys.example.json` 复制后填写），字段含义：

| 字段 | 说明 |
|------|------|
| `DASHSCOPE_API_KEY` | 阿里百炼，用于 qwen 视觉 / wan 与 qwen 图像模型 |
| `DOUBAO_API_KEY` | 火山引擎，用于 doubao-seedream 图像模型 |
| `HUNYUAN_API_KEY` / `HUNYUAN_SECRET_ID` / `HUNYUAN_SECRET_KEY` | 腾讯混元（可只填其一，缺失时自动跳过该模型） |
| `DOUBAO_IMAGE_URL` / `DOUBAO_IMAGE_MODEL` / `DOUBAO_IMAGE_FALLBACK_MODEL` / `DOUBAO_ENDPOINT_ID` | 豆包图像接口地址、主模型、降级模型、账户 Endpoint |
| `WAN_IMAGE_MODEL` / `QWEN_IMAGE_MODEL` / `QWEN_IMAGE_EDIT_MAX_MODEL` | 各图像模型名 |
| `ESP_SCREEN_ENABLED` | `"1"` 开启推屏，`"0"` 关闭 |
| `ESP_SCREEN_IP` / `ESP_SCREEN_PORT` / `ESP_SCREEN_ENDPOINT` | 小屏地址，默认 `80` + `/img` |
| `ESP_SCREEN_SIZE` / `ESP_SCREEN_JPEG_QUALITY` | 默认 `320x240`、`85` |
| `ESP_SCREEN_TIMEOUT_SEC` | 单次推送超时，默认 `3` 秒 |

没有配置某个供应商的密钥时，该模型会被自动跳过，不影响其他模型运行。

小屏的 IP 也可以在手机端右上角**设置**里改，或在电脑显示页调用 `POST /api/esp_config`，改完立即生效，无需重启。

---

## 页面与接口

| 路径 | 说明 |
|------|------|
| `/`、`/mobile` | 手机拍照 / 诊断 / 选项 / 结果，四步向导 |
| `/display` | 电脑显示页：诊断报告、候选图对比、推屏 |
| `/qrcode` | 手机页与显示页的局域网二维码 |
| `POST /api/diagnose`（别名 `POST /upload`） | 上传 + 视觉诊断 |
| `POST /api/generate` | 组装提示词 + 多模型生成 + 自动推屏（需 `generation_confirmed: true` 与 `session_id`） |
| `GET /api/latest_result` | 显示页轮询的状态快照 |
| `POST /api/push_latest_to_esp` | 手动挑选候选图推屏 |
| `GET` / `POST /api/esp_config` | 读取 / 保存小屏 IP |
| `/uploads/<path>` | 上传图片的静态访问 |

---

## 用到的模型（全部国产）

- **视觉诊断**（按顺序降级）：`glm-5.3-flash`（主力，智谱）→ `qwen3.5-flash` → `deepseek-v4-flash-vision-exp` → `qwen3-vl-plus` → `hy-vision-2.0-instruct`（腾讯混元）
- **诊断双模式**：标准（质量优先，约 30 秒）/ 快速（关思考，约 5–10 秒）
- **图像编辑**（并发比较）：`wan2.7-image`（最快）、`doubao-seedream-5.0`（人像真实感强）、`qwen-image-2.0`（生图编辑二合一）

比较模式默认开启，因此一次"生成"最多产生 3–4 次付费图像调用；未配置密钥的供应商会被跳过。结果先按画幅、清晰度、曝光、明显伪影做技术筛选，学生/老师仍可手动改选主图。

---

## 自检

以下命令只校验本地逻辑与模拟载荷，**不会发起付费请求**：

```powershell
py -3 -m unittest discover -s tests -v
py -3 -m compileall -q app main.py geometry.py
```

---

## 推送到 GitHub

远端：<https://github.com/zjq-CN/Photo-Advisor>（分支 `main`）

### 方式一：能直连 github.com 时，用常规 git

```powershell
git add -A
git commit -m "feat: ..."
git push origin main
```

### 方式二：github.com 被阻断时，用脚本走 API

国内网络下 `github.com:443` 经常连不上，但 `api.github.com` 通常可用。`tools/push_to_github.ps1` 不经过 git 传输协议，而是通过 GitHub 的 Git Data API 逐层写入对象、再移动分支指针，因此只要 API 通就能推。

**准备**：把 Personal Access Token 写进 `P.A\令牌.txt`（首行整串，该文件已被 `.gitignore` 忽略），或设置环境变量 `GITHUB_TOKEN`。令牌需要 classic PAT 的 `repo` 权限，或 fine-grained PAT 的 `Contents: Read and write`。

**先看要推什么**：

```powershell
powershell -ExecutionPolicy Bypass -File tools\push_to_github.ps1 -DryRun
```

**确认后正式推送**：

```powershell
powershell -ExecutionPolicy Bypass -File tools\push_to_github.ps1
```

脚本会做的事：

1. 读取远端 `main` 当前指向，并断言它是本地 HEAD 的祖先——**历史分叉时直接拒绝，不会强推覆盖**；
2. 对 `远端..HEAD` 的每个提交，用 `git cat-file blob` 取**规范化内容**（不是磁盘上的 CRLF 版本）上传为 blob；
3. 以父提交的 tree 为 `base_tree` 创建新 tree，并校验它与本地 `HEAD^{tree}` 一致；
4. 创建 commit（提交信息取原始 commit 对象的字节，避免多行信息被转成数组）；
5. 更新 `refs/heads/main`，成功后同步本地 `origin/main` 跟踪引用。

**已知的坑**（脚本已处理，手工推送时才需要留意）：

- 仓库 `core.autocrlf=true`，直接读磁盘文件会得到 CRLF 内容，与 git 存的 LF blob 不一致 → 必须用 `git cat-file blob`。
- PowerShell 里 `git log --format=%B` 返回的是字符串数组，直接序列化会让 GitHub 报 `422 ... is not a string` → 用 `git cat-file commit` 的原始字节按空行切分。
- GitHub 会把提交时区归一化为 `+0000`，因此同内容算出的 sha 可能与本地不同。

**常见错误**

| 现象 | 原因与处理 |
|------|-----------|
| `401 Bad credentials` | 令牌失效或复制时带了空格，换新令牌写入 `令牌.txt` |
| `403 Resource not accessible by personal access token` | 令牌权限不足，需要 `repo`（classic）或 `Contents: Read and write`（fine-grained） |
| `404 Not Found` | 仓库名/分支名写错，或令牌无权访问该仓库 |
| `Could not resolve host: github.com` | 只影响 `git push`；改用方式二 |
| 脚本提示"历史已分叉" | 先 `git fetch` 查看差异，手工合并后再推，不要强推 |

---

## 变更记录

见 [CHANGELOG.md](CHANGELOG.md)。

---

## 说明

本项目是一个高中研究性学习课题的工程实现，代码为课堂演示与答辩服务。当前分支已把早期的单文件版本（3202 行）拆分为 `app/` 包，`legacy/` 下保留了重构前的原始文件与回滚快照，供对照。
