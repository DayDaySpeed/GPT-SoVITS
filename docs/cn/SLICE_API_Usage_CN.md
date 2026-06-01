# Slice API 使用说明（跨机器：上传音频，返回 zip）

文件入口：`slice_api.py`

**不依赖服务器路径**。其它电脑通过 HTTP 上传任意格式音频，服务端一次性返回 zip（内含多段 `.wav`）。

处理流程：上传文件 → 转 WAV → 降噪 → 静音切句 → 返回 `segments.zip`。

---

## 1. 启动

```bash
python slice_api.py -a 0.0.0.0 -p 9882
```

---

## 2. 调用

```bash
curl -X POST "http://<server-ip>:9882/split_audio" \
  -F "file=@/path/to/input.mp3" \
  -F "slice_workers=1" \
  -o segments.zip
```

```bash
curl -X POST "http:127.0.0.1:9882/split_audio" \
  -F "file=@/path/to/input.mp3" \
  -F "slice_workers=1" \
  -o segments.zip
```

解压后得到：`seg_0001.wav`, `seg_0002.wav`, ...

---

## 3. Python 客户端示例

```python
import io
import zipfile
import requests

url = "http://192.168.1.10:9882/split_audio"
with open("input.mp3", "rb") as f:
    resp = requests.post(
        url,
        files={"file": ("input.mp3", f, "application/octet-stream")},
        data={"slice_workers": "4"},
        timeout=1800,
    )
resp.raise_for_status()

with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
    zf.extractall("segments_out")
print("done")
```

---

## 4. 参数

- `file`：音频文件（必填，支持 mp3/wav/flac/m4a 等）
- `slice_workers`：切片并行数，默认 `1`

---

## 5. 注意

- 同步长任务，客户端请设置较长超时（如 1800 秒）。
- 响应为二进制 zip，不包含服务器本地路径。

