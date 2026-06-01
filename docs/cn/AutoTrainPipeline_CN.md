# AutoTrain Pipeline 说明

脚本：`tools/auto_train_pipeline.py`

## 运行前：激活 Python 环境

建议所有命令都在同一个环境下执行，避免出现 `ModuleNotFoundError`（例如 `soundfile`）。

### 方式 A：venv（推荐）

在仓库根目录执行：

```bash
# 只需创建一次
python -m venv .venv

# 每次新开终端后先激活
source .venv/bin/activate

# 安装依赖（按项目实际方式）
python -m pip install -U pip
python -m pip install -r requirements.txt
```

激活后，提示符前通常会出现 `(.venv)`。

退出环境：

```bash
deactivate
```

### 方式 B：conda（与项目 README 一致）

```bash
# 只需创建一次
conda create -n GPTSoVits python=3.10 -y

# 每次新开终端后先激活
conda activate GPTSoVits

# 安装依赖（按项目实际方式）
python -m pip install -U pip
python -m pip install -r requirements.txt
```

退出环境：

```bash
conda deactivate
```

如果你已经按项目文档创建过环境，通常只需要这一条：

```bash
conda activate GPTSoVits
```

### 自检（建议每次跑训练/推理前执行）

```bash
which python
python -V
python -c "import soundfile; print('soundfile ok')"
```

如果上面最后一条失败，先安装：

```bash
python -m pip install soundfile
```

## 目录规则

如果输入是 `data/voice_input/aaa/`，则：

- prepare 产物：`data/voice_prepare/aaa/`
- train 产物：`data/voice_train/aaa/`

任务名 `aaa` 由以下规则自动推断：

1. 有 `--audio-path`：取最后一级目录名（文件则取文件名去后缀）
2. 否则有 `--reviewed-list`：若位于 `data/voice_prepare/<name>/...`，取 `<name>`

不再需要 `--exp-name`。

## 阶段

- `prepare`：转 WAV -> 降噪 -> 切片 -> ASR 生成 `.list`
- `train`：读取 `.list`，提特征并训练 GPT/SoVITS
- `all`：先跑 prepare，再跑 train

## 常用命令

### 1) prepare

```bash
python tools/auto_train_pipeline.py \
  --stage prepare \
  --audio-path data/voice_input/aaa \
  --version v2 \
  --gpu-id 0 \
  --preset auto
```

输出标注示例（默认即已清洗）：

`data/voice_prepare/aaa/asr_opt/sliced.cleaned.list`

### 2) train

```bash
python tools/auto_train_pipeline.py \
  --stage train \
  --reviewed-list data/voice_prepare/aaa_v2/asr_opt/sliced.cleaned.list \
  --version v2 \
  --gpu-id 0 \
  --preset auto
```

训练产物在：

`data/voice_train/aaa/`

### 3) all

```bash
python tools/auto_train_pipeline.py \
  --stage all \
  --audio-path data/voice_input/aaa \
  --version v2 \
  --gpu-id 0 \
  --preset auto \
  --slice-workers 4 \
  --asr-lang zh
```

## 产物结构

```text
data/voice_prepare/aaa/
├── raw_audio/
├── prepared_wav/
├── denoised/
├── sliced/
├── asr_opt/
└── .pipeline_state.json

data/voice_train/aaa/
├── 2-name2text.txt
├── 4-cnhubert/
├── 5-wav32k/
├── 6-name2semantic.tsv
├── logs_s1_v2/
└── logs_s2_v2/
```

## 主要参数

- `--audio-path`：prepare/all 必填，用于推断任务名
- `--reviewed-list`：train 必填，建议传 `data/voice_prepare/<name>/asr_opt/*.list`
- `--slice-workers`：切片并行数
- `--asr-lang`：ASR 语言（中文建议 `zh`）

## 使用生成的模型推理

训练完成后，终端会打印两条关键路径：

- `最新 GPT 权重: ...`
- `最新 SoVITS 权重: ...`

直接复制这两条路径用于推理即可。  
如果没保留终端输出，也可以手动找最新模型文件：

- `GPT_weights_v2/*.ckpt`（或你训练版本对应目录）
- `SoVITS_weights_v2/*.pth`

例如任务名是 `aaa_v2`，常见文件名：

- `GPT_weights_v2/aaa_v2-e12.ckpt`
- `SoVITS_weights_v2/aaa_v2_e20_s1240.pth`

推理命令示例：

```bash
python GPT_SoVITS/inference_cli.py \
  --gpt_model GPT_weights_v2/yizhijiu_v2_train_v3-e12.ckpt \
  --sovits_model SoVITS_weights_v2/yizhijiu_v2_train_v3_e20_s1240.pth \
  --ref_audio data/voice_ref/ref.wav \
  --ref_text data/voice_ref/ref.txt \
  --ref_language 中文 \
  --target_text data/voice_ref/target.txt \
  --target_language 中文 \
  --output_path data/voice_ref/ref_out/ \
  --output_name 早上好.wav \
  --line_by_line
```

输出文件：

- `data/voice_ref/ref_out/爱.wav`

也支持直接传完整文件路径（不需要 `--output_name`）：

```bash
python GPT_SoVITS/inference_cli.py ... --output_path data/voice_ref/ref_out/hh.wav
```

`--line_by_line` 会按 `target.txt` 的非空行逐行合成，并在行间按 `--pause_second` 拼接，适合长文/诗词减少句号附近偶发吞字。建议搭配 `--how_to_cut none` 使用。