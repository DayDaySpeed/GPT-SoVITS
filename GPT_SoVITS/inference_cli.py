import argparse
import os
import numpy as np
import soundfile as sf

from tools.i18n.i18n import I18nAuto

# 与 inference_webui 使用同一语言，避免切句模式字符串对不上
os.environ.setdefault("language", "zh_CN")
i18n = I18nAuto(language=os.environ["language"])

# CLI 别名 -> inference_webui 内部 i18n 键（与 WebUI「怎么切」一致）
HOW_TO_CUT_KEYS = {
    "none": "不切",
    "cut4": "凑四句一切",
    "cut50": "凑50字一切",
    "period_zh": "按中文句号。切",
    "period_en": "按英文句号.切",
    "punct": "按标点符号切",
}
HOW_TO_CUT_CHOICES = list(HOW_TO_CUT_KEYS.keys()) + list(HOW_TO_CUT_KEYS.values())

# 《长恨歌》等中文长诗推荐默认：
# target.txt 已「一行一联、空行分隔」时用 none（不切），按行合成整联；
# 勿用 punct，否则会在逗号处切成半句，句尾更易吞字。
DEFAULTS_LONG_POEM_ZH = {
    "how_to_cut": "none",
    "top_k": 15,
    "top_p": 1.0,
    "temperature": 0.8,
    "pause_second": 0.35,
    "speed": 1.0,
}


def resolve_how_to_cut(value: str) -> str:
    key = HOW_TO_CUT_KEYS.get(value, value)
    if key not in HOW_TO_CUT_KEYS.values():
        raise ValueError(
            f"无效的 --how_to_cut: {value!r}，可选: {', '.join(HOW_TO_CUT_CHOICES)}"
        )
    return i18n(key)


def synthesize(
    GPT_model_path,
    SoVITS_model_path,
    ref_audio_path,
    ref_text_path,
    ref_language,
    target_text_path,
    target_language,
    output_path,
    output_name="output.wav",
    how_to_cut="none",
    top_k=15,
    top_p=1.0,
    temperature=0.8,
    pause_second=0.35,
    speed=1.0,
    line_by_line=False,
):
    os.environ["gpt_path"] = GPT_model_path
    os.environ["sovits_path"] = SoVITS_model_path
    os.environ.setdefault("language", "zh_CN")
    from GPT_SoVITS.inference_webui import change_gpt_weights, change_sovits_weights, get_tts_wav

    with open(ref_text_path, "r", encoding="utf-8") as file:
        ref_text = file.read()

    with open(target_text_path, "r", encoding="utf-8") as file:
        target_text = file.read()

    change_gpt_weights(gpt_path=GPT_model_path)
    change_sovits_weights(sovits_path=SoVITS_model_path)

    how_to_cut_resolved = resolve_how_to_cut(how_to_cut)
    print(
        f"推理参数: how_to_cut={how_to_cut} -> {how_to_cut_resolved!r}, "
        f"top_k={top_k}, top_p={top_p}, temperature={temperature}, "
        f"pause_second={pause_second}, speed={speed}"
    )

    # 兼容两种写法：
    # 1) --output_path 传目录 + --output_name 指定文件名
    # 2) --output_path 直接传完整 wav 文件路径
    if output_path.lower().endswith(".wav"):
        output_wav_path = output_path
        output_dir = os.path.dirname(output_wav_path) or "."
    else:
        output_dir = output_path
        output_wav_path = os.path.join(output_dir, output_name)
    os.makedirs(output_dir, exist_ok=True)

    if line_by_line:
        lines = [line.strip() for line in target_text.splitlines() if line.strip()]
        if not lines:
            print("target_text 没有可用的非空行，未生成音频。")
            return

        merged_audio = []
        merged_sr = None
        for idx, line in enumerate(lines, start=1):
            synthesis_result = get_tts_wav(
                ref_wav_path=ref_audio_path,
                prompt_text=ref_text,
                prompt_language=i18n(ref_language),
                text=line,
                text_language=i18n(target_language),
                how_to_cut=how_to_cut_resolved,
                top_k=top_k,
                top_p=top_p,
                temperature=temperature,
                pause_second=pause_second,
                speed=speed,
            )
            result_list = list(synthesis_result)
            if not result_list:
                print(f"[WARN] 第 {idx} 行未生成音频，已跳过。")
                continue
            sr, audio_data = result_list[-1]
            if merged_sr is None:
                merged_sr = sr
            if sr != merged_sr:
                raise ValueError(f"采样率不一致: 第 {idx} 行 {sr}Hz，首行 {merged_sr}Hz")
            merged_audio.append(audio_data)
            if idx < len(lines):
                silence = np.zeros(int(merged_sr * pause_second), dtype=audio_data.dtype)
                merged_audio.append(silence)

        if not merged_audio:
            print("逐行合成未得到有效音频，未写出文件。")
            return
        final_audio = np.concatenate(merged_audio)
        sf.write(output_wav_path, final_audio, merged_sr)
        print(f"Audio saved to {output_wav_path} (line_by_line, lines={len(lines)})")
    else:
        synthesis_result = get_tts_wav(
            ref_wav_path=ref_audio_path,
            prompt_text=ref_text,
            prompt_language=i18n(ref_language),
            text=target_text,
            text_language=i18n(target_language),
            how_to_cut=how_to_cut_resolved,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            pause_second=pause_second,
            speed=speed,
        )
        result_list = list(synthesis_result)
        if result_list:
            last_sampling_rate, last_audio_data = result_list[-1]
            sf.write(output_wav_path, last_audio_data, last_sampling_rate)
            print(f"Audio saved to {output_wav_path}")


def main():
    d = DEFAULTS_LONG_POEM_ZH
    parser = argparse.ArgumentParser(
        description="GPT-SoVITS 命令行推理（默认参数针对中文长诗/长文如《长恨歌》优化）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--gpt_model", required=True, help="GPT 权重路径 (.ckpt)")
    parser.add_argument("--sovits_model", required=True, help="SoVITS 权重路径 (.pth)")
    parser.add_argument("--ref_audio", required=True, help="参考音频（建议 3~10 秒，与 ref_text 一致）")
    parser.add_argument("--ref_text", required=True, help="参考文本文件")
    parser.add_argument(
        "--ref_language", required=True, choices=["中文", "英文", "日文"], help="参考音频语种"
    )
    parser.add_argument("--target_text", required=True, help="待合成文本文件")
    parser.add_argument(
        "--target_language",
        required=True,
        choices=["中文", "英文", "日文", "中英混合", "日英混合", "多语种混合"],
        help="待合成文本语种",
    )
    parser.add_argument(
        "--output_path",
        required=True,
        help="输出路径。可传目录，或直接传 .wav 文件完整路径",
    )
    parser.add_argument(
        "--output_name",
        default="output.wav",
        help="当 --output_path 为目录时生效，指定输出文件名",
    )
    parser.add_argument(
        "--line_by_line",
        action="store_true",
        help="逐行合成 target_text 非空行并拼接，减少长文本句尾吞字",
    )

    parser.add_argument(
        "--how_to_cut",
        default=d["how_to_cut"],
        choices=HOW_TO_CUT_CHOICES,
        help=(
            "切句方式。别名: none/不切, cut4/凑四句一切, cut50/凑50字一切, "
            "period_zh/按中文句号。切, period_en/按英文句号.切, punct/按标点符号切。"
            "《长恨歌》式「一行一联+空行」文本用 none；整段无换行时用 period_zh；"
            "不要用 punct（会在逗号处切断）。"
        ),
    )
    parser.add_argument("--top_k", type=int, default=d["top_k"], help="GPT 采样 top_k（与训练配置 15 一致较稳）")
    parser.add_argument("--top_p", type=float, default=d["top_p"], help="GPT 采样 top_p")
    parser.add_argument(
        "--temperature",
        type=float,
        default=d["temperature"],
        help="GPT 采样温度；句尾易吞字时可降到 0.6~0.8",
    )
    parser.add_argument(
        "--pause_second",
        type=float,
        default=d["pause_second"],
        help="句间静音秒数（长诗朗读可略增到 0.35~0.45）",
    )
    parser.add_argument("--speed", type=float, default=d["speed"], help="语速，1.0 为正常")

    args = parser.parse_args()

    synthesize(
        args.gpt_model,
        args.sovits_model,
        args.ref_audio,
        args.ref_text,
        args.ref_language,
        args.target_text,
        args.target_language,
        args.output_path,
        output_name=args.output_name,
        how_to_cut=args.how_to_cut,
        top_k=args.top_k,
        top_p=args.top_p,
        temperature=args.temperature,
        pause_second=args.pause_second,
        speed=args.speed,
        line_by_line=args.line_by_line,
    )


if __name__ == "__main__":
    main()
