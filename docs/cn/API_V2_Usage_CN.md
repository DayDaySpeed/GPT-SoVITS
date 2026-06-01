# API v2 使用说明（HTTP 调用）

文件入口：`api_v2.py`

适用场景：将 GPT-SoVITS 作为独立 TTS 服务，被其他项目（例如 `ai_clock`）通过 HTTP 调用。

---

## 1. 启动服务

在仓库根目录执行：

```bash
python api_v2.py -a 127.0.0.1 -p 9880 -c GPT_SoVITS/configs/my_tts_infer.yaml
```

参数说明：

- `-a, --bind_addr`：监听地址，默认 `127.0.0.1`
- `-p, --port`：端口，默认 `9880`
- `-c, --tts_config`：配置文件路径，默认 `GPT_SoVITS/configs/tts_infer.yaml`

---

## 2. 主要接口

### 2.1 文本转语音：`POST /tts`

URL：

`http://127.0.0.1:9880/tts`

请求头：

- `Content-Type: application/json`

成功响应：

- 直接返回音频流（HTTP 200）

失败响应：

- 返回 JSON 错误信息（HTTP 4xx）

---

### 2.2 切换 GPT 权重：`GET /set_gpt_weights`

示例：

```bash
curl "http://127.0.0.1:9880/set_gpt_weights?weights_path=GPT_weights_v2/your_model.ckpt"
```

---

### 2.3 切换 SoVITS 权重：`GET /set_sovits_weights`

示例：

```bash
curl "http://127.0.0.1:9880/set_sovits_weights?weights_path=SoVITS_weights_v2/your_model.pth"
```

---

### 2.4 服务控制：`GET/POST /control`

- `restart`：重启
- `exit`：退出

示例：

```bash
curl "http://127.0.0.1:9880/control?command=restart"
```

---

## 3. `/tts` 请求参数说明（逐项详解）

下面按“功能分组”解释每个参数，方便你按场景调参。

### 3.1 文本与语言参数（核心）

- `text` (str，必填)
  - 作用：要合成的目标文本（LLM 回复内容）。
  - 建议：长度较长时尽量带标点，便于切句和韵律控制。
- `text_lang` (str，必填)
  - 作用：`text` 的语言标识。
  - 常用：`zh` / `en` / `ja`。
  - 建议：和文本实际语言一致，混用会影响发音。
- `prompt_text` (str，默认 `""`)
  - 作用：参考音频对应文本（音色/韵律对齐提示）。
  - 建议：尽量和 `ref_audio_path` 内容一致，不一致会降低稳定性。
- `prompt_lang` (str，必填)
  - 作用：`prompt_text` 的语言。
  - 建议：与参考音频文本语言一致。

### 3.2 参考音频参数（音色）

- `ref_audio_path` (str，必填)
  - 作用：主参考音频路径。
  - 建议：3~10 秒、干净、无背景音乐，音色越稳定效果越好。
- `aux_ref_audio_paths` (list，默认 `None`)
  - 作用：辅助参考音频列表，用于多参考融合。
  - 何时用：主参考太短或单条表现不稳定时再尝试。

### 3.3 采样与语速参数（自然度/稳定性）

- `top_k` (int，默认 `15`)
  - 作用：限制采样候选数。
  - 调整建议：增大更活泼但更随机；减小更稳。
- `top_p` (float，默认 `1`)
  - 作用：核采样阈值。
  - 调整建议：常用 `0.8~1.0`；越小越保守。
- `temperature` (float，默认 `1`)
  - 作用：采样温度，控制随机性。
  - 调整建议：句尾不稳可尝试降到 `0.6~0.9`。
- `repetition_penalty` (float，默认 `1.35`)
  - 作用：重复惩罚，降低重复字词风险。
  - 调整建议：出现“重复念”时可略升；过高可能变生硬。
- `speed_factor` (float，默认 `1.0`)
  - 作用：语速控制。
  - 常用范围：`0.8~1.2`。小于 1 变慢，大于 1 变快。

### 3.4 切句与批处理参数（你问的重点）

- `text_split_method` (str，默认 `cut5`)
  - 作用：决定长文本如何切成多段再合成。
  - 可选值：`cut0`~`cut5`（由` text_segmentation_method.py` 注册）。
  - 含义：
    - `cut0`：不切
    - `cut1`：凑四句一切
    - `cut2`：凑约 50 字一切
    - `cut3`：按中文句号 `。` 切
    - `cut4`：按英文句号 `.` 切
    - `cut5`：按标点切（中英文标点）
  - 使用建议：
    - 中文长文本默认优先试 `cut5`
    - 内容本身已手工分句时可试 `cut0`
    - 英文段落可试 `cut4`
- `batch_size` (int，默认 `1`)
  - 作用：一次并行推理多少文本片段（切句后生效）。
  - 影响：
    - 更大：吞吐可能更高，但显存/内存占用上升
    - 更小：更稳、更省资源，延迟可能略高
  - 建议：
    - 先从 `1` 开始
    - 机器资源充足再试 `2/4`
    - RK3588 或边缘设备通常建议 `1`
- `batch_threshold` (float，默认 `0.75`)
  - 作用：批处理切分阈值，影响分桶策略。
  - 建议：先保持默认，只有批处理不稳定再调。
- `split_bucket` (bool，默认 `true`)
  - 作用：是否按长度分桶再批处理。
  - 建议：开启通常更稳，除非你在排查特殊问题。
- `parallel_infer` (bool，默认 `true`)
  - 作用：并行推理开关。
  - 建议：资源吃紧或异常时可关掉做对比。

### 3.5 流式与分片参数（实时对话场景）

- `streaming_mode` (bool/int，默认 `false`)
  - 作用：是否分片流式返回音频。
  - 可用值：`false/0`（关闭）、`true/1`、`2`、`3`。
  - 经验：
    - `1/true`：质量优先，响应慢
    - `2`：中等质量，中等速度
    - `3`：更快，质量相对低
- `fragment_interval` (float，默认 `0.3`)
  - 作用：音频片段间隔控制。
  - 建议：实时交互可适当减小，避免停顿感过强。
- `overlap_length` (int，默认 `2`)
  - 作用：流式语义 token 重叠长度，缓解片段拼接断裂。
  - 建议：默认即可，听感断裂时再小步调。
- `min_chunk_length` (int，默认 `16`)
  - 作用：流式最小 chunk 长度，影响每片大小和返回频率。
  - 建议：太小可能碎片化，太大则首包变慢。

### 3.6 编码与可复现参数

- `media_type` (str，默认 `wav`)
  - 作用：返回音频编码格式。
  - 常用：`wav`（最通用）、`ogg`、`aac`。
- `seed` (int，默认 `-1`)
  - 作用：随机种子，控制可复现性。
  - 建议：要复现结果时指定固定整数。
- `sample_steps` (int，默认 `32`)
  - 作用：VITS v3 采样步数。
  - 建议：仅在 v3 相关场景调节，其他版本一般保持默认。
- `super_sampling` (bool，默认 `false`)
  - 作用：VITS v3 超采样开关。
  - 建议：仅 v3 场景按需开启，通常会增加计算开销。

---

## 4. `curl` 调用示例

```bash
curl -X POST "http://127.0.0.1:9880/tts" \
  -H "Content-Type: application/json" \
  -o data/tmp/api/my_tts_test.wav \
  -d '{
    "text": "我爱玩元神",
    "text_lang": "zh",
    "ref_audio_path": "data/voice_ref/ref.wav",
    "prompt_lang": "zh",
    "prompt_text": "我好像忘记是什么味道的了,怎么办啊,嗯,怎么办啊",
    "text_split_method": "cut0",
    "batch_size": 1,
    "top_k": 15,
    "top_p": 1.0,
    "temperature": 0.8,
    "speed_factor": 1.0,
    "fragment_interval": 0.35,
    "media_type": "wav",
    "streaming_mode": false
  }'
```

验证输出文件：

```bash
ls -lh data/tmp/api/my_tts_test.wav
```

---

## 5. 给 `ai_clock` 的集成建议

- `ai_clock` 保留业务主流程：录音 -> ASR -> LLM
- 拿到 LLM 回复文本后，调用 `POST /tts` 得到 wav
- 本地播放器播放 wav
- TTS 服务建议常驻（`systemd` 或容器）并单独监控

---

## 6. 句尾吞字/断句问题排查（推荐）

如果出现“前半句正常，后半句丢失”或“句尾被吞”，通常不是训练模型本身问题，而是推理切句和采样参数引起。

推荐先用下面这组稳态参数：

```json
{
  "text_split_method": "cut0",
  "batch_size": 1,
  "top_k": 15,
  "top_p": 1.0,
  "temperature": 0.8,
  "speed_factor": 1.0,
  "fragment_interval": 0.35,
  "streaming_mode": false
}
```

排查顺序：

1. 先把 `text_split_method` 从 `cut5` 改为 `cut0`
2. 保持 `batch_size=1`，避免并行批处理干扰
3. 把 `temperature` 下调到 `0.7~0.8`
4. 关闭流式（`streaming_mode=false`）
5. 对输入文本做清洗（合并连续标点、去除异常符号）

实战建议：

- 对话场景优先 `cut0`
- 业务层按 `。！？` 手动分句，不要按 `，` 拆句
- 每次送入 TTS 的句子尽量控制在 80~120 字内

---

## 7. RK3588 `systemd` 开机自启（推荐）

仓库已提供示例：`systemd/gpt-sovits-api.service`

使用步骤：

```bash
# 1) 拷贝服务文件
sudo cp systemd/gpt-sovits-api.service /etc/systemd/system/

# 2) 按你的实际路径修改服务文件中的 WorkingDirectory/ExecStart
sudo nano /etc/systemd/system/gpt-sovits-api.service

# 3) 重新加载并启动
sudo systemctl daemon-reload
sudo systemctl enable --now gpt-sovits-api

# 4) 查看状态和日志
systemctl status gpt-sovits-api
journalctl -u gpt-sovits-api -f
```

