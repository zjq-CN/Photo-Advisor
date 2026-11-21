# 函数迁移表（全阶段终态，脚本生成）

源：`testV1.0_backup_prompt.py`（3201 行，85 个顶层函数） → `app/` 包。

状态列说明：
- `OK`：与源文件逐字节相同（阶段① 原样搬迁，引擎断言）
- `D10改写`：`global HUNYUAN_*` 值拷贝陷阱强制的属性化改写（阶段①，行为等价论证见阶段①报告）
- `阶段②改写`：配置读取缓存化（消灭热路径读盘与 powershell 子进程）
- `阶段③拆分`：函数体按语义段落拆为多个短函数，入口保留原名
- `阶段③改写`：仅把内联的重复循环替换为共享实现，分支条件与调用序不变

## 一、85 个原始函数

| 旧行号 | 函数 | 新文件 | 新行号 | 状态 | 备注 |
|---|---|---|---|---|---|
| 50 | `load_api_keys_file` | app/config.py | 64 | 阶段②改写 | 配置缓存化 |
| 67 | `update_api_keys_file` | app/config.py | 118 | 阶段②改写 | 配置缓存化 |
| 85 | `load_env` | app/config.py | 139 | 阶段②改写 | 配置缓存化 |
| 110 | `parse_bool` | app/config.py | 154 | OK |  |
| 121 | `parse_int` | app/config.py | 165 | OK |  |
| 128 | `parse_float` | app/config.py | 172 | OK |  |
| 135 | `is_valid_ipv4` | app/config.py | 179 | OK |  |
| 143 | `get_runtime_esp_config` | app/config.py | 187 | OK |  |
| 382 | `save_latest_result` | app/storage.py | 18 | OK |  |
| 390 | `validate_session_id` | app/storage.py | 26 | OK |  |
| 397 | `build_upload_filename` | app/storage.py | 33 | OK |  |
| 410 | `save_session_data` | app/storage.py | 46 | OK |  |
| 417 | `load_session_data` | app/storage.py | 53 | OK |  |
| 426 | `open_browser_later` | app/net.py | 16 | OK |  |
| 440 | `allowed_file` | app/storage.py | 61 | OK |  |
| 444 | `image_file_to_data_url` | app/storage.py | 65 | OK |  |
| 458 | `image_file_to_jimeng_base64` | app/imagegen/jimeng.py | 20 | OK |  |
| 472 | `_repair_json_string` | app/parsing.py | 12 | OK |  |
| 505 | `extract_json_from_text` | app/parsing.py | 45 | OK |  |
| 534 | `safe_resp_text` | app/parsing.py | 74 | OK |  |
| 541 | `is_busy_or_rate_limited` | app/parsing.py | 81 | OK |  |
| 557 | `parse_qwen_image_edit_url` | app/parsing.py | 97 | OK |  |
| 573 | `parse_image_url_any` | app/parsing.py | 113 | OK |  |
| 595 | `parse_image_base64_any` | app/parsing.py | 135 | OK |  |
| 618 | `parse_task_id_any` | app/parsing.py | 158 | OK |  |
| 636 | `normalize_doubao_url` | app/imagegen/doubao.py | 13 | OK |  |
| 659 | `build_jimeng_req_key_candidates` | app/imagegen/jimeng.py | 34 | OK |  |
| 675 | `list_doubao_model_ids` | app/imagegen/doubao.py | 36 | OK |  |
| 708 | `build_doubao_model_candidates` | app/imagegen/doubao.py | 69 | OK |  |
| 739 | `sha256_hex` | app/imagegen/jimeng.py | 50 | OK |  |
| 745 | `hmac_sha256` | app/imagegen/jimeng.py | 56 | OK |  |
| 749 | `build_volc_signed_headers` | app/imagegen/jimeng.py | 60 | OK |  |
| 809 | `post_jimeng_signed` | app/imagegen/jimeng.py | 120 | OK |  |
| 830 | `save_base64_image_to_uploads` | app/storage.py | 79 | OK |  |
| 850 | `load_image_from_url_for_quality` | app/quality.py | 13 | OK |  |
| 872 | `parse_screen_size` | app/esp.py | 13 | OK |  |
| 884 | `resolve_image_ref_to_bytes` | app/esp.py | 25 | OK |  |
| 910 | `build_screen_jpeg` | app/esp.py | 51 | OK |  |
| 936 | `push_image_to_esp` | app/esp.py | 77 | OK |  |
| 974 | `build_esp_push_queue` | app/esp.py | 151 | OK |  |
| 1002 | `evaluate_image_quality` | app/quality.py | 35 | OK |  |
| 1081 | `get_local_ip` | app/net.py | 34 | OK |  |
| 1157 | `generate_qr_image` | app/net.py | 110 | OK |  |
| 1169 | `generate_access_qrcodes` | app/net.py | 122 | OK |  |
| 1184 | `build_analysis_prompt` | app/prompts.py | 11 | OK |  |
| 1216 | `normalize_ai_result` | app/parsing.py | 176 | OK |  |
| 1240 | `call_vision_once` | app/vision.py | 19 | OK |  |
| 1301 | `call_vision_hunyuan_once` | app/vision.py | 80 | OK | D10改写 |
| 1350 | `call_vision_deepseek_once` | app/vision.py | 129 | OK |  |
| 1409 | `call_vision_zhipu_once` | app/vision.py | 188 | OK |  |
| 1458 | `check_vision_model_ready` | app/vision.py | 237 | OK | D10改写 |
| 1468 | `call_vision_auto` | app/vision.py | 247 | OK |  |
| 1502 | `normalize_user_intent` | app/contracts.py | 150 | OK |  |
| 1569 | `intent_option_label` | app/contracts.py | 217 | OK |  |
| 1574 | `infer_automatic_goal` | app/strategy.py | 9 | OK |  |
| 1598 | `build_strategy_plan` | app/strategy.py | 153 | 阶段③拆分 | 128 → 23 行 |
| 1728 | `build_targeted_ideal_prompt` | app/prompts.py | 44 | OK |  |
| 1746 | `build_guidance_text` | app/strategy.py | 178 | OK |  |
| 1779 | `build_edit_prompt` | app/prompts.py | 62 | OK |  |
| 1871 | `build_generation_settings` | app/strategy.py | 211 | OK |  |
| 1884 | `edit_with_dashscope_image_model` | app/imagegen/dashscope.py | 13 | OK |  |
| 1944 | `edit_with_wan27_image` | app/imagegen/dashscope.py | 73 | OK |  |
| 1958 | `edit_with_qwen_image_20` | app/imagegen/dashscope.py | 87 | OK |  |
| 1972 | `edit_with_qwen_image_edit` | app/imagegen/dashscope.py | 101 | OK |  |
| 1986 | `edit_with_zhipu_image` | app/imagegen/zhipu.py | 11 | OK |  |
| 2010 | `edit_with_doubao_seedream` | app/imagegen/doubao.py | 100 | OK |  |
| 2115 | `poll_jimeng_result` | app/imagegen/jimeng.py | 141 | OK |  |
| 2180 | `edit_with_jimeng_image` | app/imagegen/jimeng.py | 206 | OK |  |
| 2240 | `edit_with_hunyuan_image` | app/imagegen/hunyuan.py | 28 | OK |  |
| 2397 | `get_image_model_runner` | app/imagegen/__init__.py | 17 | OK |  |
| 2406 | `check_image_model_ready` | app/imagegen/__init__.py | 26 | OK | D10改写 |
| 2424 | `get_ready_models` | app/imagegen/__init__.py | 44 | OK |  |
| 2433 | `run_model_compare` | app/imagegen/__init__.py | 118 | 阶段③拆分 | 128 → 52 行 |
| 2563 | `edit_image_auto` | app/imagegen/__init__.py | 218 | OK |  |
| 2587 | `root` | app/routes/pages.py | 14 | OK |  |
| 2596 | `mobile_page` | app/routes/pages.py | 23 | OK |  |
| 2605 | `display_page` | app/routes/pages.py | 32 | OK |  |
| 2610 | `qrcode_page` | app/routes/pages.py | 37 | OK |  |
| 2617 | `api_esp_config` | app/routes/api.py | 26 | OK |  |
| 2666 | `api_diagnose` | app/routes/api.py | 75 | OK |  |
| 2806 | `api_generate` | app/routes/api.py | 215 | 阶段③拆分 | 236 → 31 行 |
| 3044 | `upload` | app/routes/upload.py | 9 | OK |  |
| 3049 | `api_push_latest_to_esp` | app/routes/api.py | 248 | 阶段③改写 | 内联推送循环 → esp.push_queue(stop_on_first_failure=False) |
| 3145 | `api_latest_result` | app/routes/api.py | 337 | OK |  |
| 3176 | `uploaded_file` | app/routes/pages.py | 44 | OK |  |

合计 **85** 个函数，与源文件 `85` 个一致。

## 二、阶段③新增函数（拆分产物）

| 文件 | 行号 | 函数 | 行数 |
|---|---|---|---|
| app/config.py | 38 | `_api_keys_stamp` | 7 |
| app/config.py | 47 | `_read_api_keys_file_uncached` | 15 |
| app/config.py | 73 | `invalidate_api_keys_cache` | 4 |
| app/config.py | 79 | `_windows_user_env` | 37 |
| app/esp.py | 115 | `push_queue` | 34 |
| app/generation.py | 26 | `prepare_generate_request` | 54 |
| app/generation.py | 82 | `build_generating_payload` | 30 |
| app/generation.py | 114 | `_run_compare` | 33 |
| app/generation.py | 149 | `run_generation_models` | 29 |
| app/generation.py | 180 | `push_generated` | 21 |
| app/generation.py | 203 | `apply_to_session` | 22 |
| app/generation.py | 227 | `build_latest_payload` | 30 |
| app/generation.py | 259 | `build_response` | 28 |
| app/imagegen/__init__.py | 53 | `_valid_image_url` | 7 |
| app/imagegen/__init__.py | 62 | `_attempt_model` | 54 |
| app/imagegen/__init__.py | 172 | `_empty_compare_result` | 9 |
| app/imagegen/__init__.py | 183 | `_select_compare_result` | 33 |
| app/strategy.py | 119 | `_build_base_strategy` | 32 |

合计 **18** 个新增函数。

## 三、文件行数（P0：单文件 ≤400 行）

| 文件 | 行数 | P0 |
|---|---|---|
| app/__init__.py | 5 | ✅ |
| app/__main__.py | 37 | ✅ |
| app/config.py | 269 | ✅ |
| app/contracts.py | 249 | ✅ |
| app/esp.py | 176 | ✅ |
| app/generation.py | 286 | ✅ |
| app/imagegen/__init__.py | 236 | ✅ |
| app/imagegen/dashscope.py | 112 | ✅ |
| app/imagegen/doubao.py | 202 | ✅ |
| app/imagegen/hunyuan.py | 182 | ✅ |
| app/imagegen/jimeng.py | 263 | ✅ |
| app/imagegen/zhipu.py | 32 | ✅ |
| app/net.py | 131 | ✅ |
| app/parsing.py | 194 | ✅ |
| app/prompts.py | 121 | ✅ |
| app/quality.py | 108 | ✅ |
| app/routes/__init__.py | 1 | ✅ |
| app/routes/api.py | 365 | ✅ |
| app/routes/pages.py | 46 | ✅ |
| app/routes/upload.py | 11 | ✅ |
| app/server.py | 20 | ✅ |
| app/storage.py | 96 | ✅ |
| app/strategy.py | 221 | ✅ |
| app/vision.py | 275 | ✅ |

合计 24 个文件 / 3638 行。全部达标。
