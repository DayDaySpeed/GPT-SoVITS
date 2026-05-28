#!/usr/bin/env python3
"""
GPT-SoVITS 命令行全自动训练流水线。

三阶段（--stage）：
  prepare  原始音频 -> 转 WAV -> 降噪 -> 切片(可选) -> ASR 生成 .list，供人工校对
  train    校对后的 .list -> 特征提取 -> GPT/SoVITS 训练 -> 导出权重
  all      等同 prepare + train；会清空本次任务目录后重跑

目录约定（prepare/train 分目录）：
  data/voice_prepare/<name>/  预处理与 ASR 产物（切片、list、pipeline_state）
  data/voice_train/<name>/    训练相关产物（2-name2text、6-name2semantic、logs_s1_*、logs_s2_*）
  GPT_weights_* / SoVITS_weights_*  导出的推理权重

智能跳过：各步骤在 .pipeline_state.json 中记录输入签名，输入未变且输出有效则跳过。
任务名默认由 --audio-path 最后一级目录（或文件名去后缀）自动确定，例如 data/voice_input/aaa -> name=aaa。
"""
import argparse
import importlib.util
import json
import os
import re
import shutil
import time
import subprocess
import sys
from pathlib import Path

import yaml

VERSION_CHOICES = ["v1", "v2", "v3", "v4", "v2Pro", "v2ProPlus"]


ROOT = Path(__file__).resolve().parents[1]
# prepare 与 train 使用不同根目录
PREPARE_ROOT = ROOT / "data" / "voice_prepare"
TRAIN_ROOT = ROOT / "data" / "voice_train"
TMP_ROOT = ROOT / "data" / "tmp"  # 训练时临时写入的 s1.yaml / s2.json


class UserFacingError(Exception):
    """可展示给用户的配置/依赖类错误，main() 会捕获并友好退出。"""


# 各版本对应的预训练底座（GPT s1 + SoVITS s2G）
VERSION_PRETRAIN = {
    "v1": {
        "s1": "GPT_SoVITS/pretrained_models/s1bert25hz-2kh-longer-epoch=68e-step=50232.ckpt",
        "s2g": "GPT_SoVITS/pretrained_models/s2G488k.pth",
    },
    "v2": {
        "s1": "GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt",
        "s2g": "GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s2G2333k.pth",
    },
    "v3": {
        "s1": "GPT_SoVITS/pretrained_models/s1v3.ckpt",
        "s2g": "GPT_SoVITS/pretrained_models/s2Gv3.pth",
    },
    "v4": {
        "s1": "GPT_SoVITS/pretrained_models/s1v3.ckpt",
        "s2g": "GPT_SoVITS/pretrained_models/gsv-v4-pretrained/s2Gv4.pth",
    },
    "v2Pro": {
        "s1": "GPT_SoVITS/pretrained_models/s1v3.ckpt",
        "s2g": "GPT_SoVITS/pretrained_models/v2Pro/s2Gv2Pro.pth",
    },
    "v2ProPlus": {
        "s1": "GPT_SoVITS/pretrained_models/s1v3.ckpt",
        "s2g": "GPT_SoVITS/pretrained_models/v2Pro/s2Gv2ProPlus.pth",
    },
}

# 训练结束后 half/full 权重导出目录名（相对项目根）
VERSION_WEIGHT_DIR = {
    "v1": ("GPT_weights", "SoVITS_weights"),
    "v2": ("GPT_weights_v2", "SoVITS_weights_v2"),
    "v3": ("GPT_weights_v3", "SoVITS_weights_v3"),
    "v4": ("GPT_weights_v4", "SoVITS_weights_v4"),
    "v2Pro": ("GPT_weights_v2Pro", "SoVITS_weights_v2Pro"),
    "v2ProPlus": ("GPT_weights_v2ProPlus", "SoVITS_weights_v2ProPlus"),
}


def run_cmd(cmd, env=None):
    print("[CMD]", " ".join(str(i) for i in cmd))
    subprocess.run(cmd, cwd=ROOT, check=True, env=env)


def run_cmd_with_retry(cmd, env=None, retries: int = 3, base_delay_sec: float = 3.0, retry_label: str = ""):
    """ffmpeg / 降噪 / ASR 等易受 IO 或网络影响，失败时指数退避重试。"""
    last_error = None
    for attempt in range(1, retries + 1):
        try:
            run_cmd(cmd, env=env)
            return
        except subprocess.CalledProcessError as e:
            last_error = e
            if attempt >= retries:
                break
            delay = base_delay_sec * (2 ** (attempt - 1))
            label = retry_label or "命令"
            print(f"[WARN] {label} 失败，第 {attempt}/{retries} 次，{delay:.0f}s 后重试...")
            time.sleep(delay)
    raise last_error


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def audio_files_in_dir(path: Path):
    if not path.exists() or not path.is_dir():
        return []
    exts = {".wav", ".mp3", ".flac", ".m4a", ".ogg", ".aac", ".wma"}
    return sorted([p for p in path.iterdir() if p.is_file() and p.suffix.lower() in exts])


def file_has_content(path: Path) -> bool:
    return path.exists() and path.is_file() and path.stat().st_size > 0


def load_pipeline_state(exp_dir: Path) -> dict:
    """读取 <exp_dir>/.pipeline_state.json，用于判断某步是否可跳过。"""
    state_file = exp_dir / ".pipeline_state.json"
    if not state_file.exists():
        return {}
    try:
        return json.loads(state_file.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_pipeline_state(exp_dir: Path, state: dict):
    state_file = exp_dir / ".pipeline_state.json"
    state_file.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def audio_signature(paths: list[Path]) -> list[dict]:
    """用路径+大小+mtime 描述输入集；输入变化则对应步骤需重跑。"""
    sig = []
    for p in sorted(paths, key=lambda x: str(x)):
        if not p.exists() or not p.is_file():
            continue
        st = p.stat()
        sig.append(
            {
                "path": str(p.resolve()),
                "size": st.st_size,
                "mtime_ns": st.st_mtime_ns,
            }
        )
    return sig


def clear_audio_outputs(folder: Path):
    if not folder.exists():
        return
    for p in folder.iterdir():
        if p.is_file():
            p.unlink(missing_ok=True)


def count_nonempty_lines(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def count_files_with_ext(folder: Path, suffix: str) -> int:
    if not folder.exists():
        return 0
    return sum(1 for p in folder.glob(f"*{suffix}") if p.is_file())


def infer_task_name(audio_path: Path | None, reviewed_list: Path | None) -> str:
    """
    推断任务名：
    1) prepare/all：优先取 --audio-path 的最后一级目录名（或文件 stem）
    2) train：若 list 位于 data/voice_prepare/<name>/...，取该 <name>
    3) 无法推断时报错提示用户提供可推断路径
    """
    if audio_path is not None:
        p = audio_path.resolve()
        if p.is_dir():
            return p.name
        return p.stem
    if reviewed_list is not None:
        rp = reviewed_list.resolve()
        for parent in rp.parents:
            if parent.parent == PREPARE_ROOT:
                return strip_version_suffix(parent.name)
    raise UserFacingError("无法推断任务名：请提供 --audio-path（prepare/all）或有效的 --reviewed-list（train）")


def version_suffix(version: str) -> str:
    return f"_{version.lower()}"


def strip_version_suffix(name: str) -> str:
    # 兼容历史目录名：aaa_v2 / aaa_v2pro / aaa_v2proplus / aaa_v3 / aaa_v4
    return re.sub(r"_(v1|v2|v3|v4|v2pro|v2proplus)$", "", name, flags=re.IGNORECASE)


def infer_version_from_task_name(name: str) -> str | None:
    m = re.search(r"_(v1|v2|v3|v4|v2pro|v2proplus)$", name, flags=re.IGNORECASE)
    if not m:
        return None
    token = m.group(1).lower()
    mapping = {
        "v1": "v1",
        "v2": "v2",
        "v3": "v3",
        "v4": "v4",
        "v2pro": "v2Pro",
        "v2proplus": "v2ProPlus",
    }
    return mapping[token]


def next_duplicate_task_name(task_base_name: str, resolved_version: str) -> str:
    """
    在已有 <name>_vX 基础上生成下一个不冲突名称：
    aaa_v2 -> aaa1_v2 -> aaa2_v2 ...
    """
    suffix = version_suffix(resolved_version)
    idx = 1
    while True:
        candidate = f"{task_base_name}{idx}{suffix}"
        if not (PREPARE_ROOT / candidate).exists():
            return candidate
        idx += 1


def ask_prepare_conflict_action(existing_prepare_dir: Path) -> str:
    """
    prepare 发现目录已存在时，询问用户：
    1 覆盖重跑；2 新建重复版本目录。
    """
    print(f"[WARN] prepare 输出目录已存在: {existing_prepare_dir}")
    print("[INPUT] 请选择操作：")
    print("  1) 覆盖重跑（清空现有目录）")
    print("  2) 新建目录重跑（如 aaa1_v2 / aaa2_v2）")
    while True:
        try:
            choice = input("请输入 1 或 2: ").strip()
        except EOFError:
            raise UserFacingError("检测到已存在 prepare 目录，但当前环境无法交互选择。请在终端交互运行。")
        if choice in {"1", "2"}:
            return choice
        print("输入无效，请输入 1 或 2。")


def assert_reviewed_list_in_work_dir(list_path: Path, work_dir: Path):
    try:
        list_path.resolve().relative_to(work_dir.resolve())
    except ValueError:
        raise UserFacingError(
            f"--reviewed-list 须位于 prepare 目录下并与本次任务一致:\n"
            f"  工作目录: {work_dir}\n"
            f"  当前 list: {list_path}"
        )


def check_python_module(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def preflight_check(stage: str):
    """按阶段检查 ffmpeg 与 Python 依赖，缺失时给出 conda/pip 安装提示。"""
    missing = []
    if shutil.which("ffmpeg") is None:
        missing.append(("ffmpeg", "conda install -n GPTSoVits ffmpeg"))

    if stage in {"prepare", "all"}:
        if not check_python_module("modelscope"):
            missing.append(("modelscope", 'python -m pip install "modelscope[audio]"'))
        if not check_python_module("addict"):
            missing.append(("addict", "python -m pip install addict"))
        if not check_python_module("faster_whisper"):
            missing.append(("faster-whisper", "python -m pip install faster-whisper"))

    if stage in {"train", "all"}:
        if not check_python_module("soundfile"):
            missing.append(("soundfile", "python -m pip install soundfile"))

    if missing:
        lines = ["启动前依赖检查失败，缺少以下依赖："]
        for name, _ in missing:
            lines.append(f"- {name}")
        lines.append("")
        lines.append("请先在 GPTSoVits 环境安装：")
        lines.append("conda activate GPTSoVits")
        for _, cmd in missing:
            lines.append(cmd)
        raise UserFacingError("\n".join(lines))


def convert_audio_to_wav(input_path: Path, output_path: Path):
    # 项目内训练/推理统一为单声道 32kHz WAV
    run_cmd_with_retry(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(input_path),
            "-ac",
            "1",
            "-ar",
            "32000",
            str(output_path),
        ],
        retries=2,
        retry_label=f"音频转 WAV ({input_path.name})",
    )


def prepare_wav_inputs(source_dir: Path, prepared_dir: Path):
    ensure_dir(prepared_dir)
    supported = {".wav", ".mp3", ".flac", ".m4a", ".ogg", ".aac", ".wma"}
    for src in sorted(source_dir.iterdir()):
        if not src.is_file():
            continue
        ext = src.suffix.lower()
        if ext not in supported:
            continue
        dst = prepared_dir / f"{src.stem}.wav"
        if file_has_content(dst):
            print(f"[SKIP] WAV 已存在: {dst}")
            continue
        if ext == ".wav":
            if src.resolve() != dst.resolve():
                shutil.copy2(src, dst)
        else:
            convert_audio_to_wav(src, dst)
    return prepared_dir


def count_audio_minutes(list_file: Path) -> float:
    """从 .list 第一列 wav 路径统计总时长，供 auto 预设调 epoch/batch。"""
    import soundfile as sf  # 仅 train/all 需要，延迟导入避免 prepare 阶段强依赖

    total_sec = 0.0
    with list_file.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            wav_path = line.split("|", 1)[0]
            info = sf.info(wav_path)
            total_sec += float(info.frames) / float(info.samplerate)
    return total_sec / 60.0


def normalize_text(text: str) -> str:
    text = text.strip()
    text = re.sub(r"\s+", "", text)
    text = text.replace("。", ".").replace("，", ",").replace("！", "!").replace("？", "?")
    return text


def is_text_low_quality(text: str, min_len: int, max_len: int) -> tuple[bool, str]:
    if not text:
        return True, "empty_text"
    if len(text) < min_len:
        return True, "too_short_text"
    if len(text) > max_len:
        return True, "too_long_text"
    if re.search(r"(.)\1{4,}", text):
        return True, "repeat_chars"
    cjk = len(re.findall(r"[\u4e00-\u9fff]", text))
    letters = len(re.findall(r"[A-Za-z]", text))
    punct = len(re.findall(r"[,.!?，。！？、…：:;；\-~～]", text))
    valid = cjk + letters + punct
    if valid == 0:
        return True, "invalid_chars"
    if cjk / max(1, len(text)) < 0.45:
        return True, "low_cjk_ratio"
    return False, ""


def clean_training_list(input_list: Path, output_list: Path, report_json: Path) -> Path:
    """
    过滤低质量切片：过短/过长音频、空文本、重复文本、非中文占比过低等。
    输出 <stem>.cleaned.list 与 <stem>.cleaned.report.json（最多记录 200 条剔除样本）。
    """
    import soundfile as sf

    allow_lang = {"ZH", "zh"}
    min_seconds = 1.2
    max_seconds = 15.0
    min_text_len = 4
    max_text_len = 80

    kept = []
    removed = []
    seen_text = set()
    lines = input_list.read_text(encoding="utf-8").splitlines()

    for idx, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        parts = line.split("|")
        if len(parts) < 4:
            removed.append({"line": idx, "reason": "bad_format", "raw": line})
            continue
        wav_path, spk, lang, text = parts[0], parts[1], parts[2], "|".join(parts[3:])
        wav = Path(wav_path)
        if not wav.exists():
            removed.append({"line": idx, "reason": "missing_wav", "wav": wav_path})
            continue
        if lang not in allow_lang:
            removed.append({"line": idx, "reason": "lang_not_allowed", "lang": lang, "wav": wav_path})
            continue
        try:
            info = sf.info(str(wav))
            dur = info.frames / info.samplerate
        except Exception:
            removed.append({"line": idx, "reason": "bad_audio", "wav": wav_path})
            continue
        if dur < min_seconds:
            removed.append({"line": idx, "reason": "too_short_audio", "seconds": dur, "wav": wav_path})
            continue
        if dur > max_seconds:
            removed.append({"line": idx, "reason": "too_long_audio", "seconds": dur, "wav": wav_path})
            continue

        text = normalize_text(text)
        bad, reason = is_text_low_quality(text, min_text_len, max_text_len)
        if bad:
            removed.append({"line": idx, "reason": reason, "wav": wav_path, "text": text})
            continue
        if text in seen_text:
            removed.append({"line": idx, "reason": "dup_text", "wav": wav_path, "text": text})
            continue
        seen_text.add(text)
        kept.append(f"{wav_path}|{spk}|{lang}|{text}")

    output_list.parent.mkdir(parents=True, exist_ok=True)
    output_list.write_text("\n".join(kept) + ("\n" if kept else ""), encoding="utf-8")

    stats = {
        "input_total": len(lines),
        "kept_total": len(kept),
        "removed_total": len(removed),
        "kept_ratio": round(len(kept) / max(1, len(lines)), 4),
        "removed_samples": removed[:200],
    }
    report_json.parent.mkdir(parents=True, exist_ok=True)
    report_json.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[INFO] 清洗完成: 保留 {len(kept)} / {len(lines)}")
    print(f"[INFO] 清洗列表: {output_list}")
    print(f"[INFO] 清洗报告: {report_json}")
    return output_list


def auto_hparams(total_minutes: float):
    """数据越少 epoch 越多；数据越多适当减少 epoch 防过拟合，并增大 batch。"""
    if total_minutes < 5:
        return {"gpt_epochs": 20, "sovits_epochs": 30, "batch": 4}
    if total_minutes < 15:
        return {"gpt_epochs": 16, "sovits_epochs": 24, "batch": 6}
    if total_minutes < 40:
        return {"gpt_epochs": 12, "sovits_epochs": 20, "batch": 8}
    return {"gpt_epochs": 8, "sovits_epochs": 12, "batch": 12}


def resolve_train_hparams(preset: str, total_minutes: float, s1_template: dict, s2_template: dict):
    """preset=auto 按时长；webui-default 原样采用模板 yaml/json 里的 epoch 与 batch。"""
    if preset == "webui-default":
        return {
            "gpt_epochs": int(s1_template["train"]["epochs"]),
            "sovits_epochs": int(s2_template["train"]["epochs"]),
            "s1_batch": int(s1_template["train"]["batch_size"]),
            "s2_batch": int(s2_template["train"]["batch_size"]),
            "name": "webui-default",
        }
    auto = auto_hparams(total_minutes)
    return {
        "gpt_epochs": int(auto["gpt_epochs"]),
        "sovits_epochs": int(auto["sovits_epochs"]),
        "s1_batch": int(auto["batch"]),
        "s2_batch": int(auto["batch"]),
        "name": "auto",
    }


def discover_latest_weight(folder: Path, suffix: str) -> Path | None:
    if not folder.exists():
        return None
    files = [p for p in folder.glob(f"*{suffix}") if p.is_file()]
    if not files:
        return None
    return sorted(files, key=lambda p: p.stat().st_mtime)[-1]


def resolve_audio_input(src: Path, exp_dir: Path, do_slice: bool, slice_workers: int):
    """
    预处理音频并返回最终用于 ASR 的目录。

    固定顺序：raw -> prepared_wav(32k mono) -> denoised -> sliced(可选)。
    每步对比 .pipeline_state 与输出目录，输入未变则跳过。
    """
    if not src.exists():
        raise FileNotFoundError(f"音频路径不存在: {src}")
    raw_dir = exp_dir / "raw_audio"
    ensure_dir(raw_dir)
    if src.is_file():
        dst = raw_dir / src.name
        if dst.resolve() != src.resolve():
            shutil.copy2(src, dst)
        source_dir = raw_dir
        input_sig = audio_signature([src])
    else:
        source_dir = src
        input_sig = audio_signature(audio_files_in_dir(source_dir))

    state = load_pipeline_state(exp_dir)

    # 固定先统一转为 wav（含 mp3 自动转 wav）
    prepared_dir = exp_dir / "prepared_wav"
    prepare_state = {
        "input_sig": input_sig,
    }
    prepared_current = audio_signature(audio_files_in_dir(prepared_dir))
    if state.get("prepare_wav") == prepare_state and prepared_current:
        print(f"[SKIP] WAV 转换已完成且输入未变化: {prepared_dir}")
    else:
        clear_audio_outputs(prepared_dir)
        source_dir = prepare_wav_inputs(source_dir, prepared_dir)
        state["prepare_wav"] = prepare_state
        save_pipeline_state(exp_dir, state)
    source_dir = prepared_dir

    # 固定先降噪
    denoised_dir = exp_dir / "denoised"
    ensure_dir(denoised_dir)
    src_audio = audio_files_in_dir(source_dir)
    denoise_state = {
        "prepared_sig": audio_signature(src_audio),
    }
    denoised_audio = audio_files_in_dir(denoised_dir)
    if state.get("denoise") == denoise_state and len(denoised_audio) >= len(src_audio) and len(src_audio) > 0:
        print(f"[SKIP] 降噪已完成且输入未变化: {denoised_dir}")
    else:
        clear_audio_outputs(denoised_dir)
        run_cmd_with_retry(
            [
                sys.executable,
                "tools/cmd-denoise.py",
                "-i",
                str(source_dir),
                "-o",
                str(denoised_dir),
            ],
            retries=3,
            retry_label="降噪",
        )
        state["denoise"] = denoise_state
        save_pipeline_state(exp_dir, state)
    source_dir = denoised_dir

    if not do_slice:
        return source_dir

    sliced_dir = exp_dir / "sliced"
    ensure_dir(sliced_dir)
    slice_state = {
        "denoised_sig": audio_signature(audio_files_in_dir(source_dir)),
        "slice_workers": slice_workers,
        # 与 tools/slice_audio.py 命令行参数一致（见下方 run_cmd 中的数字串）
        "slice_params": {
            "threshold": -34,
            "min_length": 4000,
            "min_interval": 300,
            "hop_size": 20,
            "max_sil_kept": 500,
            "max": 0.9,
            "alpha": 0.25,
        },
    }
    sliced_audio = audio_files_in_dir(sliced_dir)
    if state.get("slice") == slice_state and sliced_audio:
        print(f"[SKIP] 切片已完成且输入未变化: {sliced_dir}")
    else:
        clear_audio_outputs(sliced_dir)
        workers = max(1, slice_workers)
        for i in range(workers):
            run_cmd(
                [
                    sys.executable,
                    "tools/slice_audio.py",
                    str(source_dir),
                    str(sliced_dir),
                    "-34",
                    "4000",
                    "300",
                    "20",
                    "500",
                    "0.9",
                    "0.25",
                    str(i),
                    str(workers),
                ]
            )
        state["slice"] = slice_state
        save_pipeline_state(exp_dir, state)
    return sliced_dir


def generate_list_with_asr(exp_dir: Path, audio_dir: Path, asr_out_dir: Path, asr_model: str, asr_lang: str, precision: str):
    """对 audio_dir 跑 faster-whisper，生成 asr_opt/<目录名>.list（wav|spk|lang|text）。"""
    ensure_dir(asr_out_dir)
    list_file = asr_out_dir / f"{audio_dir.name}.list"
    state = load_pipeline_state(exp_dir)
    asr_state = {
        "audio_sig": audio_signature(audio_files_in_dir(audio_dir)),
        "asr_model": asr_model,
        "asr_lang": asr_lang,
        "precision": precision,
    }
    if state.get("asr") == asr_state and file_has_content(list_file):
        print(f"[SKIP] ASR 标注已存在且输入未变化: {list_file}")
        return list_file
    if list_file.exists():
        list_file.unlink(missing_ok=True)
    run_cmd_with_retry(
        [
            sys.executable,
            "tools/asr/fasterwhisper_asr.py",
            "-i",
            str(audio_dir),
            "-o",
            str(asr_out_dir),
            "-s",
            asr_model,
            "-l",
            asr_lang,
            "-p",
            precision,
        ],
        retries=3,
        retry_label="ASR 转写",
    )
    if not list_file.exists():
        raise FileNotFoundError(f"ASR 未生成标注文件: {list_file}")
    state["asr"] = asr_state
    save_pipeline_state(exp_dir, state)
    return list_file


def run_dataset_prepare(work_name: str, list_file: Path, wav_dir: str, gpu_id: str, version: str, is_half: bool):
    """
    调用官方 1-get-text / 2-get-hubert-wav32k / 3-get-semantic（及 Pro 系 2-get-sv）。
    产物写在 data/voice_train/<work_name>/，供后续 s1/s2 训练读取。
    """
    exp_dir = TRAIN_ROOT / work_name
    ensure_dir(exp_dir)
    expected_samples = count_nonempty_lines(list_file)
    if expected_samples <= 0:
        raise UserFacingError(f"标注文件为空或无有效行: {list_file}")

    env = os.environ.copy()
    env.update(
        {
            "inp_text": str(list_file),
            "inp_wav_dir": wav_dir,
            "exp_name": work_name,
            "i_part": "0",
            "all_parts": "1",
            "_CUDA_VISIBLE_DEVICES": gpu_id,
            "opt_dir": str(exp_dir),
            "bert_pretrained_dir": "GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large",
            "cnhubert_base_dir": "GPT_SoVITS/pretrained_models/chinese-hubert-base",
            "is_half": str(is_half),
            "s2config_path": "GPT_SoVITS/configs/s2.json"
            if version not in {"v2Pro", "v2ProPlus"}
            else f"GPT_SoVITS/configs/s2{version}.json",
            "pretrained_s2G": VERSION_PRETRAIN[version]["s2g"],
        }
    )

    sem_merge = exp_dir / "6-name2semantic.tsv"
    text_merged = exp_dir / "2-name2text.txt"
    text_part = exp_dir / "2-name2text-0.txt"
    hubert_dir = exp_dir / "4-cnhubert"
    wav32_dir = exp_dir / "5-wav32k"
    sv_dir = exp_dir / "7-sv_cn"

    if file_has_content(text_merged):
        print(f"[SKIP] 文本标注已存在: {text_merged}")
    else:
        run_cmd([sys.executable, "GPT_SoVITS/prepare_datasets/1-get-text.py"], env=env)
        if text_part.exists():
            text_part.replace(text_merged)
        if not file_has_content(text_merged):
            raise UserFacingError(f"文本标注生成失败: {text_merged}")

    hubert_count = count_files_with_ext(hubert_dir, ".pt")
    wav32_count = count_files_with_ext(wav32_dir, ".wav")
    if hubert_count >= expected_samples and wav32_count >= expected_samples:
        print(f"[SKIP] Hubert 特征已存在: {hubert_dir} ({hubert_count}/{expected_samples})")
    else:
        run_cmd([sys.executable, "GPT_SoVITS/prepare_datasets/2-get-hubert-wav32k.py"], env=env)

    if "Pro" in version:
        sv_count = count_files_with_ext(sv_dir, ".pt")
        if sv_count >= expected_samples:
            print(f"[SKIP] 说话人向量已存在: {sv_dir} ({sv_count}/{expected_samples})")
        else:
            env["sv_path"] = "GPT_SoVITS/pretrained_models/sv/pretrained_eres2netv2w24s4ep4.ckpt"
            run_cmd([sys.executable, "GPT_SoVITS/prepare_datasets/2-get-sv.py"], env=env)

    if file_has_content(sem_merge):
        print(f"[SKIP] 语义标注已存在: {sem_merge}")
    else:
        run_cmd([sys.executable, "GPT_SoVITS/prepare_datasets/3-get-semantic.py"], env=env)
        sem_part = exp_dir / "6-name2semantic-0.tsv"
        if sem_part.exists():
            with sem_merge.open("w", encoding="utf-8") as f:
                f.write("item_name\tsemantic_audio\n")
                f.write(sem_part.read_text(encoding="utf-8").strip() + "\n")
        if not file_has_content(sem_merge):
            raise UserFacingError(f"语义标注生成失败: {sem_merge}")


def run_train(
    work_name: str,
    version: str,
    gpu_id: str,
    total_minutes: float,
    is_half: bool,
    preset: str,
):
    """依次训练 GPT(s1) 与 SoVITS(s2)；数据与日志均在 data/voice_train/<work_name>/。"""
    ensure_dir(TMP_ROOT)
    work_dir = TRAIN_ROOT / work_name
    ensure_dir(work_dir)
    gpt_dir, sovits_dir = VERSION_WEIGHT_DIR[version]
    ensure_dir(ROOT / gpt_dir)
    ensure_dir(ROOT / sovits_dir)

    s1_template = ROOT / ("GPT_SoVITS/configs/s1longer.yaml" if version == "v1" else "GPT_SoVITS/configs/s1longer-v2.yaml")
    with s1_template.open("r", encoding="utf-8") as f:
        s1 = yaml.safe_load(f)
    s2_template = ROOT / ("GPT_SoVITS/configs/s2.json" if version not in {"v2Pro", "v2ProPlus"} else f"GPT_SoVITS/configs/s2{version}.json")
    with s2_template.open("r", encoding="utf-8") as f:
        s2 = json.load(f)

    hp = resolve_train_hparams(preset=preset, total_minutes=total_minutes, s1_template=s1, s2_template=s2)
    # 根据显存上限预先压低 SoVITS batch，减少首次 OOM 概率
    try:
        import torch

        if torch.cuda.is_available():
            vram_gb = torch.cuda.get_device_properties(int(gpu_id)).total_memory / (1024**3)
            if vram_gb <= 8.2:
                hp["s2_batch"] = min(hp["s2_batch"], 4)
            elif vram_gb <= 10.5:
                hp["s2_batch"] = min(hp["s2_batch"], 6)
    except Exception:
        pass
    print(
        f"训练预设: {hp['name']} | GPT epochs={hp['gpt_epochs']} batch={hp['s1_batch']} | "
        f"SoVITS epochs={hp['sovits_epochs']} batch={hp['s2_batch']}"
    )

    s1["train"]["epochs"] = hp["gpt_epochs"]
    s1["train"]["batch_size"] = hp["s1_batch"]
    s1["train"]["if_save_latest"] = True
    s1["train"]["if_save_every_weights"] = True
    s1["train"]["save_every_n_epoch"] = 1
    s1["train"]["exp_name"] = work_name
    s1["train"]["half_weights_save_dir"] = gpt_dir
    work_dir_rel = str(Path("data") / "voice_train" / work_name)
    s1["train_semantic_path"] = f"{work_dir_rel}/6-name2semantic.tsv"
    s1["train_phoneme_path"] = f"{work_dir_rel}/2-name2text.txt"
    s1["output_dir"] = f"{work_dir_rel}/logs_s1_{version}"
    s1["pretrained_s1"] = VERSION_PRETRAIN[version]["s1"]
    # ASR 用 float32 时训练也走全精度，batch 减半以控制显存
    if not is_half:
        s1["train"]["precision"] = "32"
        s1["train"]["batch_size"] = max(1, s1["train"]["batch_size"] // 2)

    s1_cfg = TMP_ROOT / f"{work_name}_s1.yaml"
    with s1_cfg.open("w", encoding="utf-8") as f:
        yaml.safe_dump(s1, f, allow_unicode=True, sort_keys=False)

    env_s1 = os.environ.copy()
    env_s1["_CUDA_VISIBLE_DEVICES"] = gpu_id
    env_s1["hz"] = "25hz"
    # PyTorch >=2.6 默认会强制 torch.load(weights_only=True)，
    # 旧版 lightning checkpoint 里含有非白名单对象（如 pathlib.PosixPath）会恢复失败。
    env_s1["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
    run_cmd([sys.executable, "GPT_SoVITS/s1_train.py", "--config_file", str(s1_cfg)], env=env_s1)

    s2["train"]["epochs"] = hp["sovits_epochs"]
    s2_batch = hp["s2_batch"]
    s2["train"]["batch_size"] = s2_batch
    s2["train"]["if_save_latest"] = True
    s2["train"]["if_save_every_weights"] = True
    s2["train"]["save_every_epoch"] = 1
    s2["train"]["gpu_numbers"] = gpu_id
    s2["train"]["text_low_lr_rate"] = 0.4
    s2["train"]["pretrained_s2G"] = VERSION_PRETRAIN[version]["s2g"]
    s2d = VERSION_PRETRAIN[version]["s2g"].replace("s2G", "s2D")
    s2["train"]["pretrained_s2D"] = s2d if (ROOT / s2d).exists() else ""
    s2["model"]["version"] = version
    # 数据与 checkpoint 均使用同一工作目录
    s2["data"]["exp_dir"] = work_dir_rel
    s2["s2_ckpt_dir"] = work_dir_rel
    s2["save_weight_dir"] = sovits_dir
    s2["name"] = work_name
    s2["version"] = version
    if not is_half:
        s2["train"]["fp16_run"] = False
        s2["train"]["batch_size"] = max(1, s2["train"]["batch_size"] // 2)

    s2_cfg = TMP_ROOT / f"{work_name}_s2.json"
    ensure_dir(ROOT / work_dir_rel / f"logs_s2_{version}")
    # OOM 时自动减半 batch 重试（最多 3 次）
    for attempt in range(1, 4):
        s2["train"]["batch_size"] = s2_batch
        with s2_cfg.open("w", encoding="utf-8") as f:
            json.dump(s2, f, ensure_ascii=False, indent=2)
        try:
            run_cmd([sys.executable, "GPT_SoVITS/s2_train.py", "--config", str(s2_cfg)])
            break
        except subprocess.CalledProcessError:
            if s2_batch <= 1 or attempt >= 3:
                raise
            next_batch = max(1, s2_batch // 2)
            print(f"[WARN] SoVITS 训练失败，自动降低 batch: {s2_batch} -> {next_batch}，重试...")
            s2_batch = next_batch

    latest_gpt = discover_latest_weight(ROOT / gpt_dir, ".ckpt")
    latest_sovits = discover_latest_weight(ROOT / sovits_dir, ".pth")
    print("\n训练完成")
    print("工作目录:", work_dir)
    print("训练输出名:", work_name)
    print("最新 GPT 权重:", latest_gpt if latest_gpt else "未找到")
    print("最新 SoVITS 权重:", latest_sovits if latest_sovits else "未找到")


def parse_args():
    parser = argparse.ArgumentParser(description="GPT-SoVITS 全自动标注与训练流水线")
    parser.add_argument("--stage", choices=["prepare", "train", "all"], default="prepare")
    parser.add_argument("--audio-path", help="原始音频文件或目录")
    parser.add_argument("--reviewed-list", help="你校对后的标注文件路径(.list)")
    parser.add_argument("--version", default=None, choices=VERSION_CHOICES)
    parser.add_argument("--gpu-id", default="0", help="单卡 id，如 0")
    parser.add_argument("--no-slice", action="store_true", help="关闭切片，直接对输入目录 ASR")
    parser.add_argument("--slice-workers", type=int, default=1, help="切片并行份数")
    parser.add_argument("--asr-model", default="large-v3", help="faster-whisper 模型，如 large-v3")
    parser.add_argument("--asr-lang", default="auto", help="ASR语言，auto/zh/en/ja...")
    parser.add_argument("--asr-precision", default="float16", choices=["float16", "float32", "int8"])
    parser.add_argument(
        "--preset",
        default="auto",
        choices=["auto", "webui-default"],
        help="训练参数预设: auto=按时长自动调参, webui-default=使用WebUI默认配置",
    )
    return parser.parse_args()


def main():
    try:
        args = parse_args()
        preflight_check(args.stage)

        ensure_dir(PREPARE_ROOT)
        ensure_dir(TRAIN_ROOT)

        reviewed_for_resolve = Path(args.reviewed_list).resolve() if args.reviewed_list else None
        audio_path_for_name = Path(args.audio_path).resolve() if args.audio_path else None
        resolved_version = args.version
        if resolved_version is None:
            if args.stage == "train":
                if not reviewed_for_resolve:
                    raise UserFacingError("train 阶段未指定 --version 时，必须提供 --reviewed-list 用于自动推断版本")
                inferred_version = None
                for parent in reviewed_for_resolve.parents:
                    if parent.parent == PREPARE_ROOT:
                        inferred_version = infer_version_from_task_name(parent.name)
                        break
                if inferred_version is None:
                    raise UserFacingError("无法从 --reviewed-list 路径推断版本，请显式指定 --version")
                resolved_version = inferred_version
                print(f"[INFO] 未指定 --version，已从 reviewed-list 推断版本: {resolved_version}")
            else:
                resolved_version = "v2"
        task_base_name = infer_task_name(audio_path_for_name, reviewed_for_resolve)
        task_name = f"{task_base_name}{version_suffix(resolved_version)}"
        prepare_dir = PREPARE_ROOT / task_name
        train_dir = TRAIN_ROOT / task_name

        if args.stage == "prepare" and prepare_dir.exists():
            choice = ask_prepare_conflict_action(prepare_dir)
            if choice == "1":
                print(f"[INFO] 将覆盖重跑: {prepare_dir}")
                shutil.rmtree(prepare_dir)
            else:
                task_name = next_duplicate_task_name(task_base_name, resolved_version)
                prepare_dir = PREPARE_ROOT / task_name
                train_dir = TRAIN_ROOT / task_name
                print(f"[INFO] 已切换为新目录: {prepare_dir}")


        ensure_dir(prepare_dir)
        ensure_dir(train_dir)
        print(f"[INFO] 任务名: {task_name}")
        print(f"[INFO] prepare 目录: {prepare_dir}")
        print(f"[INFO] train 目录: {train_dir}")

        is_half = args.asr_precision == "float16"
        generated_list = None

        if args.stage in {"prepare", "all"}:
            if not args.audio_path:
                raise UserFacingError("prepare/all 阶段必须提供 --audio-path")
            audio_dir = resolve_audio_input(
                Path(args.audio_path).resolve(), prepare_dir, not args.no_slice, args.slice_workers
            )
            asr_out = prepare_dir / "asr_opt"
            raw_list = generate_list_with_asr(
                exp_dir=prepare_dir,
                audio_dir=audio_dir,
                asr_out_dir=asr_out,
                asr_model=args.asr_model,
                asr_lang=args.asr_lang,
                precision=args.asr_precision,
            )
            # prepare/all 默认直接产出清洗后的 list
            cleaned = raw_list.with_name(f"{raw_list.stem}.cleaned.list")
            report = raw_list.with_name(f"{raw_list.stem}.cleaned.report.json")
            generated_list = clean_training_list(raw_list, cleaned, report)
            raw_list.unlink(missing_ok=True)
            print("\n标注文件已生成（已清洗），请先校对：", generated_list)
            if args.stage == "prepare":
                return  # prepare 到此结束，等待用户改 .list 后再 train

        if args.stage in {"train", "all"}:
            # train 必须有人工校对后的 list；all 则直接用当次 ASR 结果（不校对）
            list_path = reviewed_for_resolve if reviewed_for_resolve else generated_list
            if not list_path or not list_path.exists():
                raise UserFacingError("train/all 阶段需要 --reviewed-list（或 all 阶段成功生成的标注）")
            if args.stage == "train":
                assert_reviewed_list_in_work_dir(list_path, prepare_dir)
            total_minutes = count_audio_minutes(list_path)
            print(f"检测数据总时长: {total_minutes:.2f} 分钟，将自动调整训练参数")
            run_dataset_prepare(
                work_name=task_name,
                list_file=list_path,
                wav_dir="",
                gpu_id=args.gpu_id,
                version=resolved_version,
                is_half=is_half,
            )
            run_train(
                work_name=task_name,
                version=resolved_version,
                gpu_id=args.gpu_id,
                total_minutes=total_minutes,
                is_half=is_half,
                preset=args.preset,
            )
    except UserFacingError as e:
        print(f"[ERROR] {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
