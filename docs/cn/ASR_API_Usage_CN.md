# ASR API 使用说明（HTTP 调用）

文件入口：`asr_api.py`

适用场景：将语音识别能力作为 HTTP 服务给其他项目调用（例如 `ai_clock`），输入音频路径，返回识别文本。

---

## 1. 启动服务

在仓库根目录执行：

```bash
python asr_api.py -a 127.0.0.1 -p 9881 -s large-v3 -l auto -pr float16
```

参数说明：

- `-a, --bind_addr`：监听地址，默认 `127.0.0.1`
- `-p, --port`：端口，默认 `9881`
- `-s, --model_size`：Faster-Whisper 模型大小，默认 `large-v3`
- `-l, --language`：默认识别语言，默认 `auto`
- `-pr, --precision`：计算精度，默认 `float16`

---

## 2. 主要接口

### 2.1 语音转文本：`GET/POST /asr`

URL：

`http://127.0.0.1:9881/asr`

成功响应（仅返回识别文本字段）：

```json
{
  "text": "早上好，今天也要加油。"
}
```

失败响应：

```json
{
  "message": "asr failed",
  "Exception": "..."
}
```

---

## 3. 请求参数

- `audio_path` (str, 必填)：待识别音频文件路径
- `language` (str, 可选)：语言代码，默认 `auto`
- `engine` (str, 可选)：`auto` / `fasterwhisper` / `funasr`，默认 `auto`
- `model_size` (str, 可选)：Whisper 模型大小，默认 `large-v3`
- `precision` (str, 可选)：`float16` / `float32` / `int8`，默认 `float16`

### `engine` 选择建议

- `auto`：默认推荐；中文/粤语优先走 FunASR，其它语种走 Faster-Whisper
- `funasr`：只用于中文/粤语
- `fasterwhisper`：多语种场景

---

## 4. 调用示例

### 4.1 GET 示例

```bash
curl "http://127.0.0.1:9881/asr?audio_path=data/voice_ref/ref.wav&language=zh&engine=auto"
```

### 4.2 POST 示例（推荐）

```bash
curl -X POST "http://127.0.0.1:9881/asr" \
  -H "Content-Type: application/json" \
  -d '{
    "audio_path": "data/voice_ref/ref.wav",
    "language": "zh",
    "engine": "auto",
    "model_size": "large-v3",
    "precision": "float16"
  }'
```

---

## 5. 服务控制

与 `api_v2.py` 一样，支持：

- `GET /control?command=restart`
- `GET /control?command=exit`

示例：

```bash
curl "http://127.0.0.1:9881/control?command=restart"
```
