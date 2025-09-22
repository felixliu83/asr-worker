# -*- coding: utf-8 -*-
import os
import io
import re
import json
import time
import wave
import uuid
import queue
import shutil
import logging
import threading
from typing import Any, Dict, List, Optional, Tuple
import subprocess, tempfile, os, shutil

import soundfile as sf
import numpy as np
from fastapi import FastAPI, File, UploadFile, Request, Query
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel

# ========= 你现有模块：尽量不改名，直接复用 =========
# - finalize_segments: words + diar_turns -> segments（内含按说话人切段、Merge、标点修正等）
# - clean_text: 文本规整
from modules.align import finalize_segments  # 保持你项目结构
from modules.text_clean import clean_text    # 保持你项目结构

# ========= 基础设置 =========
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

WORKDIR      = os.environ.get("WORKDIR", "/workspace")
DATA_DIR     = os.path.join(WORKDIR, "app", "data")
UPLOAD_DIR   = os.path.join(DATA_DIR, "uploads")
RESULT_DIR   = os.path.join(DATA_DIR, "results")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

DEVICE       = os.environ.get("DEVICE", "cuda")      # "cuda" / "cpu"
WHISPER_MODEL= os.environ.get("WHISPER_MODEL", "large-v3")
COMPUTE_TYPE = os.environ.get("COMPUTE_TYPE", "float16")  # cuda: float16, cpu: int8
DIARIZE      = os.environ.get("DIARIZE", "on")       # "on" / "off" / "auto"
PYANNOTE_TOKEN = os.environ.get("PYANNOTE_TOKEN")

SIMULATE     = os.environ.get("SIMULATE", "false").lower() == "true"

app = FastAPI()

# 任务记录（轻量内存队列）
tasks: Dict[str, Dict[str, Any]] = {}
lock = threading.Lock()


# ========= 小工具 =========

def _append_log(t: Dict[str, Any], msg: str) -> None:
    t.setdefault("logs", []).append(msg)

def _fail_task(t: Dict[str, Any], err: str) -> None:
    t["status"] = "failed"
    t["error"]  = err

def _public_url(req: Dict[str, Any], local_path: str) -> str:
    base = (req or {}).get("base_url") or ""
    # 你的 /data 是直接挂静态的：/data/results/<file>
    # 这里返回完整可访问 URL
    rel = os.path.relpath(local_path, start=os.path.join(WORKDIR, "app"))
    return f"{base}/{rel.replace(os.sep, '/')}"

def _strip_invisibles(s: str) -> str:
    if not s:
        return s
    inv = re.compile(r"[\x00-\x08\x0B-\x0C\x0E-\x1F\u200B-\u200F\u202A-\u202E\u2060-\u206F\uFEFF]")
    return inv.sub("", s)

def _run(cmd: list, timeout: int = 600):
    """运行子进程，报错时抛异常并带上stderr，便于排查。"""
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=timeout)
    if p.returncode != 0:
        raise RuntimeError(f"cmd failed: {' '.join(cmd)}\n--- stderr ---\n{p.stderr}")
    return p

def _ffprobe_has_audio(path: str) -> bool:
    """用 ffprobe 判断是否有音频流，避免喂错文件。"""
    try:
        cmd = [
            "ffprobe", "-v", "error", "-select_streams", "a:0",
            "-show_entries", "stream=codec_type", "-of", "default=noprint_wrappers=1:nokey=1",
            path,
        ]
        p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=30)
        return ("audio" in (p.stdout or "")) and p.returncode == 0
    except Exception:
        return False

def to_wav16k_mono(src_path: str) -> str:
    """
    用 ffmpeg 强制把任意格式（含 m4a/aac）转成 16k/mono 的 wav。
    返回一个临时路径（/tmp/…/audio_16k_mono.wav）；调用者只需要用路径，不要去读内存。
    """
    if not os.path.exists(src_path):
        raise FileNotFoundError(f"audio not found: {src_path}")

    if not _ffprobe_has_audio(src_path):
        # 给出更清晰的错误，方便你在 /asr/status 里定位到“文件没音频流/格式异常”
        raise ValueError(f"no audio stream or unrecognized format by ffprobe: {src_path}")

    tmpdir = tempfile.mkdtemp(prefix="wav16k_")
    dst = os.path.join(tmpdir, "audio_16k_mono.wav")

    # -y 覆盖；-vn 去视频；-sn 去字幕；-ac 1 单声道；-ar 16000 采样率；-map a:0 取第一个音轨
    cmd = [
        "ffmpeg", "-nostdin", "-hide_banner", "-y",
        "-i", src_path,
        "-vn", "-sn",
        "-ac", "1",
        "-ar", "16000",
        "-map", "a:0",
        "-loglevel", "error",
        dst,
    ]
    try:
        _run(cmd, timeout=900)
    except Exception as e:
        # 确保把临时目录清走（避免泄露），同时抛出更友好的错误
        try:
            shutil.rmtree(tmpdir)
        except Exception:
            pass
        raise RuntimeError(f"ffmpeg decode failed: {e}")

    # 最后做一道存在性校验
    if not os.path.exists(dst) or os.path.getsize(dst) == 0:
        try:
            shutil.rmtree(tmpdir)
        except Exception:
            pass
        raise RuntimeError("ffmpeg produced empty wav output")

    return dst

# ========= WhisperX 安全封装 =========

def _non_empty_diar(obj) -> bool:
    """True 当且仅当 diarization 结果存在且非空。
    兼容 pandas.DataFrame / list / tuple / dict / pyannote.Annotation。
    """
    if obj is None:
        return False
    try:
        import pandas as pd  # type: ignore
        if isinstance(obj, pd.DataFrame):
            return not obj.empty
    except Exception:
        pass
    try:
        return len(obj) > 0  # type: ignore[arg-type]
    except Exception:
        return True


def _diar_to_turns_for_finalize(diar_obj) -> List[Tuple[float, float, str]]:
    """把 WhisperX diarization 返回（DataFrame/Annotation）转换为 finalize_segments 的 turns:
       [(start, end, speaker), ...]
    """
    turns: List[Tuple[float, float, str]] = []
    if not _non_empty_diar(diar_obj):
        return turns

    # DataFrame: 列名通常包含 "start","end","speaker"
    try:
        import pandas as pd  # type: ignore
        if isinstance(diar_obj, pd.DataFrame):
            for _, row in diar_obj.iterrows():
                try:
                    ts = float(row["start"])
                    te = float(row["end"])
                    sp = str(row.get("speaker", ""))
                    turns.append((ts, te, sp))
                except Exception:
                    continue
            return turns
    except Exception:
        pass

    # pyannote Annotation
    try:
        for segment, label in diar_obj.itertracks(yield_label=True):  # type: ignore[attr-defined]
            try:
                ts = float(segment.start)
                te = float(segment.end)
                sp = str(label)
                turns.append((ts, te, sp))
            except Exception:
                continue
    except Exception:
        # 尝试 list[dict]
        try:
            for it in diar_obj:
                ts = float(it["start"]); te = float(it["end"])
                sp = str(it.get("speaker", ""))
                turns.append((ts, te, sp))
        except Exception:
            pass

    return turns


def transcribe_with_whisperx(
    audio_path: str,
    model_name: str,
    device: str = "cuda",
    compute_type: str = "float16",
    lang_hint: Optional[str] = None,
    ctx_prompt: Optional[str] = None,   # WhisperX 不支持 prompt，这里保留占位
    pyannote_token: Optional[str] = None,
    enable_diar: bool = True,
    batch_size: int = 16,
):
    """
    返回:
      words: List[Dict]  每个词 {start, end, text, (speaker)}
      asr_raw: Dict      包含 language/segments/diarization(已转 list[dict]) 等
    """
    import whisperx

    # 1) 基础 ASR
    model = whisperx.load_model(model_name, device, compute_type=compute_type)

    # 2) 语言：None 自动，否则强制
    language = None if (not lang_hint or lang_hint == "auto") else lang_hint

    # 3) 转写（不传 initial_prompt / vad）
    asr = model.transcribe(
        audio_path,
        language=language,
        task="transcribe",
        batch_size=batch_size,
    )
    detected_lang = asr.get("language") or language or "auto"
    logging.info(f"[whisperx] asr_lang={detected_lang}")

    # 4) CTC 对齐
    align_lang = detected_lang if detected_lang != "auto" else (lang_hint or "zh")
    am, meta = whisperx.load_align_model(language_code=align_lang, device=device)
    result_dict = whisperx.align(asr["segments"], am, meta, audio_path, device)

    # 标准化为 dict(segments=[...])
    if isinstance(result_dict, list):
        result_dict = {"segments": result_dict}
    elif not isinstance(result_dict, dict):
        result_dict = {"segments": []}
    result_dict.setdefault("segments", [])

    # 5) 说话人分离（WhisperX 的帧级，输出 DataFrame/Annotation）
    diar_obj = None
    if enable_diar and pyannote_token:
        logging.info("[whisperx] stage=diar_frames_start (token=yes)")
        diar = whisperx.DiarizationPipeline(use_auth_token=pyannote_token, device=device)
        diar_obj = diar(audio_path)
        shape = getattr(diar_obj, "shape", None)
        logging.info(f"[whisperx] diarize_segments type={type(diar_obj).__name__}, shape={shape}")
        if _non_empty_diar(diar_obj):
            try:
                # 这里可以把说话人直接“写入”到对齐后的 segments/words
                result_dict = whisperx.assign_word_speakers(diar_obj, result_dict)
                logging.info("[whisperx] word_speakers_assigned_by_frames")
            except Exception as e:
                logging.exception(f"[whisperx] assign_word_speakers failed: {e}")
        else:
            logging.info("[whisperx] diarization empty -> skip assign speakers")
    else:
        logging.info("[whisperx] diarization disabled or no token")

    # 6) 抽取词级（带 speaker）
    words: List[Dict[str, Any]] = []
    for seg in result_dict.get("segments", []) or []:
        spk = seg.get("speaker")
        for w in (seg.get("words") or []):
            if "word" in w and w.get("start") is not None and w.get("end") is not None:
                item = {"start": float(w["start"]), "end": float(w["end"]), "text": w["word"]}
                if spk:
                    item["speaker"] = spk
                words.append(item)

    # 7) 导出 diarization 为 list[dict]（避免 DataFrame 直接塞 JSON）
    diar_export: Optional[List[Dict[str, Any]]] = None
    if _non_empty_diar(diar_obj):
        try:
            import pandas as pd  # type: ignore
            if isinstance(diar_obj, pd.DataFrame):
                diar_export = diar_obj.to_dict(orient="records")
            else:
                # Annotation -> list[dict]
                tmp = []
                try:
                    for segm, label in diar_obj.itertracks(yield_label=True):  # type: ignore[attr-defined]
                        tmp.append({"start": float(segm.start), "end": float(segm.end), "speaker": str(label)})
                    diar_export = tmp
                except Exception:
                    diar_export = None
        except Exception:
            diar_export = None

    return words, {
        "language": detected_lang,
        "segments": result_dict.get("segments", []),
        "diarization": diar_export,
    }


# ========= 任务主流程（精简版）=========

def _process_task(task_id: str):
    # 0) 取任务
    try:
        with lock:
            t = tasks.get(task_id)
            if not t:
                return
            t["status"] = "processing"
            t["progress"] = 5
            audio_path = t.get("audio_path")
            req = t.get("request")
            ctx_prompt = t.get("context_prompt") or None
            _append_log(t, "start_processing")
            if not audio_path or not os.path.exists(audio_path):
                _fail_task(t, f"audio_missing: {audio_path}")
                return
    except Exception:
        return

    # 1) 模拟
    if SIMULATE:
        time.sleep(0.5)
        demo = [
            {"start": 0.0, "end": 3.0, "text": "模拟-欢迎", "speaker": "SPEAKER_00"},
            {"start": 3.0, "end": 7.0, "text": "模拟-讨论", "speaker": "SPEAKER_00"},
            {"start": 7.0, "end": 11.0, "text": "模拟-总结", "speaker": "SPEAKER_01"},
        ]
        srt_path  = os.path.join(RESULT_DIR, f"{task_id}.srt")
        json_path = os.path.join(RESULT_DIR, f"{task_id}.json")
        os.makedirs(RESULT_DIR, exist_ok=True)
        from modules.align import write_srt as _write_srt  # 若你已有 _write_srt，请改回你的导入
        from modules.align import save_json as _save_json  # 同上
        _write_srt(demo, srt_path)
        _save_json({"task_id": task_id, "segments": demo}, json_path)
        preview = "\n".join(((f"[{d.get('speaker')}] " if d.get("speaker") else "") + d["text"]).strip() for d in demo)
        with lock:
            t = tasks[task_id]
            t["status"] = "succeeded"
            t["progress"] = 100
            t["result"] = {
                "aligned_preview": preview,
                "srt_url": _public_url(req, srt_path),
                "json_url": _public_url(req, json_path),
            }
            _append_log(t, "done (simulate)")
        return

    # 2) 实际处理
    try:
        # 2.1 预处理
        with lock:
            t = tasks[task_id]
            _append_log(t, "stage=prepare_audio")
            t["progress"] = 10
        norm_audio = to_wav16k_mono(audio_path)
        with lock:
            t = tasks[task_id]
            _append_log(t, f"audio_prepared: {norm_audio}")
            t["progress"] = 15

        # 2.2 ASR + 对齐 + 可选说话人
        with lock:
            t = tasks[task_id]
            _append_log(t, "stage=asr_whisperx_start")
            t["progress"] = 25

        # 语言 hint：任务里有则用，没有用环境/默认
        with lock:
            t = tasks[task_id]
            lang_hint = t.get("lang_hint") or os.environ.get("LANG_HINT", "auto")
        words, asr_raw = transcribe_with_whisperx(
            norm_audio,
            model_name=WHISPER_MODEL,
            device=DEVICE,
            compute_type=COMPUTE_TYPE,
            lang_hint=lang_hint,
            ctx_prompt=ctx_prompt,  # 被忽略，仅占位
            pyannote_token=PYANNOTE_TOKEN,
            enable_diar=(DIARIZE == "on" or (DIARIZE == "auto" and PYANNOTE_TOKEN)),
            batch_size=16,
        )
        with lock:
            t = tasks[task_id]
            _append_log(t, f"asr_lang={asr_raw.get('language')}")
            t["progress"] = 40

        # 2.3 为 finalize_segments 准备 diar_turns（turns 不必须，提供更稳）
        diar_turns: List[Tuple[float, float, str]] = _diar_to_turns_for_finalize(asr_raw.get("diarization"))

        # 2.4 合段（一次性完成；遇到 speaker 变更要切段 —— 由你的 finalize_segments 负责）
        final_segments = finalize_segments(words, diar_turns, lang_hint=(lang_hint or "zh"))

        # 2.5 统一清洗（去隐形字符 + 你的 clean_text），一次即可
        for seg in final_segments:
            seg["text"] = _strip_invisibles(seg.get("text", ""))
            seg["text"] = clean_text(seg["text"], lang_hint="auto")
        # 过滤空段
        final_segments = [s for s in final_segments if s.get("text")]

        # 2.6 说话人重排（第一个开口的人 -> SPEAKER_00）
        try:
            from modules.align import renumber_speakers_by_first_appearance
            final_segments = renumber_speakers_by_first_appearance(final_segments)
        except Exception:
            pass

        # 2.7 导出
        srt_path  = os.path.join(RESULT_DIR, f"{task_id}.srt")
        json_path = os.path.join(RESULT_DIR, f"{task_id}.json")
        os.makedirs(RESULT_DIR, exist_ok=True)

        # 你项目里已有 _write_srt/_save_json 就继续用它们；这里用 align 模块里的同名函数占位
        from modules.align import write_srt as _write_srt
        from modules.align import save_json as _save_json
        _write_srt(final_segments, srt_path)
        _save_json(
            {
                "task_id": task_id,
                "segments": final_segments,
                # 附带原始片段，便于调试（可按需去掉）
                "asr_raw": {
                    "language": asr_raw.get("language"),
                    "segments": asr_raw.get("segments", []),
                    "diarization": asr_raw.get("diarization", None),
                },
            },
            json_path,
        )

        preview = "\n".join(
            ((f"[{s.get('speaker')}] " if s.get("speaker") else "") + s["text"]).strip()
            for s in final_segments[:12]
        ).strip()

        with lock:
            t = tasks[task_id]
            t["result"] = {
                "aligned_preview": preview,
                "srt_url": _public_url(t.get("request"), srt_path),
                "json_url": _public_url(t.get("request"), json_path),
            }
            t["status"] = "succeeded"
            t["progress"] = 100
            _append_log(t, f"finalize_done: segs={len(final_segments)}")
            _append_log(t, "done")

    except Exception as e:
        import traceback
        tb = traceback.format_exc(limit=3)
        with lock:
            t = tasks.get(task_id) or {}
            _fail_task(t, f"{type(e).__name__}: {e}")
            _append_log(t, "fatal: " + str(e))
            _append_log(t, tb)


# ========= HTTP API =========

@app.get("/healthz")
def healthz():
    try:
        ver_path = os.path.join(WORKDIR, "VERSION")
        version = open(ver_path).read().strip() if os.path.exists(ver_path) else "unknown"
    except Exception:
        version = "unknown"
    return {
        "ok": True,
        "model": WHISPER_MODEL,
        "device": DEVICE,
        "simulate": SIMULATE,
        "version": version,
    }


@app.post("/asr/upload")
def asr_upload(request: Request, file: UploadFile = File(...)):
    task_id = str(uuid.uuid4())

    # 1) 保存
    filename = f"{task_id}_{file.filename}"
    save_path = os.path.join(UPLOAD_DIR, filename)
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # 2) 初始化任务
    with lock:
        tasks[task_id] = {
            "status": "queued",
            "progress": 5,
            "audio_path": save_path,
            "request": {"base_url": str(request.base_url).rstrip("/")},
            "context_prompt": None,
            "result": None,
            "logs": [],
        }
        size = os.path.getsize(save_path)
        _append_log(tasks[task_id], f"upload_ok: path={save_path}, size={size}B")

    # 3) 后台线程
    threading.Thread(target=_process_task, args=(task_id,), daemon=True).start()

    # 4) 返回
    return {"task_id": task_id, "status": "queued"}


@app.get("/asr/status")
def asr_status(task_id: str = Query(...)):
    with lock:
        t = tasks.get(task_id)
        if not t:
            return JSONResponse({"task_id": task_id, "status": "not_found"}, status_code=404)
        return {
            "task_id": task_id,
            "status": t.get("status"),
            "progress": t.get("progress"),
            "logs": t.get("logs", []),
            "error": t.get("error"),
        }


@app.get("/asr/result")
def asr_result(task_id: str = Query(...)):
    with lock:
        t = tasks.get(task_id)
        if not t:
            return JSONResponse({"task_id": task_id, "status": "not_found"}, status_code=404)
        if t.get("status") != "succeeded":
            return {"task_id": task_id, "status": t.get("status")}
        res = dict(t.get("result") or {})
    # 附带 segments 到结果（方便一次拿全）
    try:
        json_path = os.path.join(RESULT_DIR, f"{task_id}.json")
        data = json.load(open(json_path, "r", encoding="utf-8"))
        res["segments"] = data.get("segments", [])
    except Exception:
        pass
    return res


# =========（可选）静态文件映射 =========
# 你项目里可能已有静态文件中间件，这里省略。
# Run: uvicorn asr_worker:app --host 0.0.0.0 --port 8000