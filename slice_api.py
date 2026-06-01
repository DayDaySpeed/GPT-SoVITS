"""
# slice_api.py

跨机器可用的单文件切分 API：
  上传任意格式音频 -> 降噪 -> 静音切句 -> 一次性返回 zip（内含多段 .wav）

` python slice_api.py -a 0.0.0.0 -p 9882 `

POST /split_audio  (multipart)
  - file: 音频文件（mp3/wav/flac/...）
  - slice_workers: 1

响应: application/zip（seg_0001.wav, seg_0002.wav, ...）
"""

import argparse
import os
import shutil
import signal
import sys
import tempfile
import zipfile
from io import BytesIO
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse, Response

now_dir = os.getcwd()
sys.path.append(now_dir)

from tools.auto_train_pipeline import (  # noqa: E402
    UserFacingError,
    audio_files_in_dir,
    ensure_dir,
    preflight_check,
    resolve_audio_input,
)

parser = argparse.ArgumentParser(description="GPT-SoVITS slice-audio api")
parser.add_argument("-a", "--bind_addr", type=str, default="127.0.0.1")
parser.add_argument("-p", "--port", type=int, default=9882)
args = parser.parse_args()

host = args.bind_addr
port = args.port
argv = sys.argv

APP = FastAPI()


def handle_control(command: str):
    if command == "restart":
        os.execl(sys.executable, sys.executable, *argv)
    elif command == "exit":
        os.kill(os.getpid(), signal.SIGTERM)
        exit(0)


def _flatten_segments(sliced_dir: Path, output_dir: Path) -> List[Path]:
    ensure_dir(output_dir)
    src_files = audio_files_in_dir(sliced_dir)
    if not src_files:
        raise UserFacingError("切片结果为空，请检查输入音频时长或内容")

    out_paths: List[Path] = []
    for idx, src in enumerate(src_files, start=1):
        dst = output_dir / f"seg_{idx:04d}.wav"
        shutil.copy2(src, dst)
        out_paths.append(dst)
    return out_paths


def split_uploaded_file(input_path: Path, slice_workers: int) -> tuple[Path, List[Path]]:
    if not input_path.is_file():
        raise UserFacingError("上传内容不是有效文件")

    job_dir = Path(tempfile.mkdtemp(prefix="gsv_split_"))
    work_dir = job_dir / "_work"
    segments_dir = job_dir / "segments"
    ensure_dir(work_dir)
    ensure_dir(segments_dir)

    try:
        sliced_dir = resolve_audio_input(
            input_path,
            work_dir,
            do_slice=True,
            slice_workers=max(1, slice_workers),
        )
        segments = _flatten_segments(sliced_dir, segments_dir)
        return job_dir, segments
    finally:
        if work_dir.exists():
            shutil.rmtree(work_dir, ignore_errors=True)


def build_zip_bytes(segment_paths: List[Path]) -> bytes:
    buf = BytesIO()
    with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_STORED) as zf:
        for p in segment_paths:
            zf.write(p, arcname=p.name)
    return buf.getvalue()


async def read_upload(file: UploadFile) -> Path:
    suffix = Path(file.filename or "audio").suffix or ".bin"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix, prefix="gsv_in_") as tf:
        content = await file.read()
        tf.write(content)
        return Path(tf.name)


@APP.get("/control")
async def control(command: str = None):
    if command is None:
        return JSONResponse(status_code=400, content={"message": "command is required"})
    handle_control(command)


@APP.post("/split_audio")
async def split_audio(
    file: UploadFile = File(...),
    slice_workers: int = Form(1),
):
    """上传单个音频，一次性返回包含多段 wav 的 zip。"""
    if not file.filename:
        return JSONResponse(status_code=400, content={"message": "file is required"})

    temp_input: Optional[Path] = None
    job_dir: Optional[Path] = None
    try:
        preflight_check("prepare")
        temp_input = await read_upload(file)
        job_dir, segments = split_uploaded_file(temp_input, slice_workers=slice_workers)
        zip_bytes = build_zip_bytes(segments)
        return Response(
            content=zip_bytes,
            media_type="application/zip",
            headers={"Content-Disposition": 'attachment; filename="segments.zip"'},
        )
    except UserFacingError as e:
        return JSONResponse(status_code=400, content={"message": str(e)})
    except Exception as e:
        return JSONResponse(status_code=400, content={"message": "split failed", "Exception": str(e)})
    finally:
        if temp_input and temp_input.exists():
            temp_input.unlink(missing_ok=True)
        if job_dir and job_dir.exists():
            shutil.rmtree(job_dir, ignore_errors=True)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(APP, host=host, port=port, workers=1)
