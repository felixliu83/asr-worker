import os, json, uuid, time, threading, torch
from typing import Dict, Any, Optional, List, Tuple
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import requests
import subprocess, shutil, tempfile, os, re
from uuid import uuid4
from modules.text_clean import clean_text
from modules.align import finalize_segments, assign_speaker_to_words, renumber_speakers_by_first_appearance
from faster_whisper import WhisperModel


BASE_DIR      = os.environ.get("BASE_DIR", "./data")
UPLOAD_DIR    = os.path.join(BASE_DIR, "uploads")
RESULT_DIR    = os.path.join(BASE_DIR, "results")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "small")
DEVICE        = os.environ.get("DEVICE", "auto")           # auto/cuda/cpu
COMPUTE_TYPE  = os.environ.get("COMPUTE_TYPE", "int8")     # cpu:int8; gpu:float16
DIARIZE       = os.environ.get("DIARIZE", "off")          # on/off/auto
PYANNOTE_TOKEN= os.environ.get("PYANNOTE_TOKEN", "")

SIMULATE      = os.environ.get("WORKER_SIMULATE", "0") == "1"
SIM_DELAY     = int(os.environ.get("SIM_DELAY_SECONDS", "2"))

app = FastAPI(title="ASR Worker (Real)")
tasks: Dict[str, Dict[str, Any]] = {}
lock = threading.Lock()

app.mount("/results", StaticFiles(directory=RESULT_DIR), name="results")

ZW_RE = re.compile(r"[\x00\u200B-\u200F\u202A-\u202E\u2060-\u206F\uFEFF]")

def _strip_invisibles(s: str) -> str:
    return ZW_RE.sub("", s or "")


def _device_for_pyannote():
    dev = (os.environ.get("DEVICE") or "auto").lower()
    if dev in ("auto", ""):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        return torch.device(dev)
    except Exception:
        return torch.device("cpu")

def _device_str_for_whisper():
    dev = (os.environ.get("DEVICE") or "auto").lower()
    if dev in ("auto", ""):
        return "cuda" if torch.cuda.is_available() else "cpu"
    # 兼容 "cuda:0" 之类写法
    return "cuda" if dev.startswith("cuda") else "cpu"

def _append_log(task, msg):
    task.setdefault("logs", []).append(str(msg))

def _fail_task(task, msg):
    task["status"] = "failed"
    _append_log(task, f"fatal: {msg}")

def _resolve_device():
    import os, torch
    dev = (os.environ.get("DEVICE") or "").lower().strip()

    # 允许 auto / cuda / cpu / gpu
    if dev in ("auto", "", None):
        return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    if dev in ("cuda", "gpu"):
        return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    if dev == "cpu":
        return torch.device("cpu")

    # 兼容 "cuda:0" 这种写法
    try:
        return torch.device(dev)
    except Exception:
        return torch.device("cpu")


def _asr_load_model():
    device = _device_str_for_whisper()  # 字符串
    compute = (os.environ.get("COMPUTE_TYPE") or "auto").lower()
    if compute in ("auto", ""):
        compute = "float16" if device == "cuda" else "int8"  # CPU 默认 int8 更稳
    return WhisperModel(WHISPER_MODEL, device=device, compute_type=compute, device_index=0)


def _fmt_srt_time(t: float) -> str:
    if t < 0: t = 0.0
    ms = int((t - int(t)) * 1000)
    s  = int(t) % 60
    m  = (int(t) // 60) % 60
    h  = int(t) // 3600
    return f"{h:02}:{m:02}:{s:02},{ms:03}"

def _write_srt(segs: List[Dict[str,Any]], path: str):
    lines = []
    for i, seg in enumerate(segs, 1):
        spk = seg.get("speaker")
        header = f"{i}\n{_fmt_srt_time(seg['start'])} --> {_fmt_srt_time(seg['end'])}\n"
        text = seg["text"].strip()
        if spk: text = f"[{spk}] {text}"
        lines.append(header + text + "\n")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

def _save_json(obj: Any, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def _public_url(request_or_base, path: str) -> str:
    # 已是完整 URL，直接返回
    if isinstance(path, str) and (path.startswith("http://") or path.startswith("https://")):
        return path

    base = ""
    obj = request_or_base

    # FastAPI Request 对象
    if hasattr(obj, "headers") or hasattr(obj, "base_url"):
        try:
            xfh = obj.headers.get("x-forwarded-host")
            xfp = obj.headers.get("x-forwarded-proto", "https")
            if xfh:
                base = f"{xfp}://{xfh}"
            else:
                base = str(obj.base_url).rstrip("/")
        except Exception:
            base = ""
    # 历史写法：dict 里可能存了 {"base_url": "..."}
    elif isinstance(obj, dict):
        base = str(obj.get("base_url", "")).rstrip("/")
    # 推荐写法：直接存字符串
    elif isinstance(obj, str):
        base = obj.rstrip("/")

    rel = path.lstrip("./")
    return f"{base}/{rel}" if base else rel
def _safe_wav_for_diar(original_path: str) -> str:
    """给 pyannote 用的 16k 单声道临时 wav；ffmpeg 不在则原样返回。"""
    if not shutil.which("ffmpeg"):
        return original_path
    try:
        tmpdir = tempfile.mkdtemp(prefix="diar_")
        out = os.path.join(tmpdir, "diar_16k_mono.wav")
        # -ac 1 单声道，-ar 16000 采样率
        subprocess.run(
            ["ffmpeg", "-y", "-i", original_path, "-ac", "1", "-ar", "16000", "-vn", out],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True
        )
        return out
    except Exception:
        return original_path


def to_wav16k_mono(src_path: str) -> str:
    """将任意音频转成 16k/mono WAV；ffmpeg 不在则原样返回。"""
    if not shutil.which("ffmpeg"):
        return src_path
    tmpdir = tempfile.mkdtemp(prefix="wav16k_")
    out = os.path.join(tmpdir, "audio_16k_mono.wav")
    cmd = ["ffmpeg", "-y", "-i", src_path, "-ac", "1", "-ar", "16000", "-vn", out]
    try:
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, check=True)
        return out
    except subprocess.CalledProcessError as e:
        # 记录 ffmpeg stderr，便于排错
        with lock:
            _append_log(tasks[task_id], f"ffmpeg_failed: {e.stderr.decode(errors='ignore')[:300]}")
        return src_path


class StartBody(BaseModel):
    audio_url: str
    series_id: Optional[str] = ""
    meeting_id: Optional[str] = ""
    context_prompt: Optional[str] = ""
    glossary: Optional[List[str]] = []

@app.get("/healthz")
def healthz():
    version = os.environ.get("APP_VERSION", "unknown")
    if version == "unknown":
        try:
            with open("/workspace/VERSION", "r", encoding="utf-8") as f:
                version = f.read().strip() or "unknown"
        except Exception:
            pass
    return {
        "ok": True,
        "model": WHISPER_MODEL,
        "device": _resolve_device(),
        "simulate": SIMULATE,
        "version": version,
    }

@app.post("/asr/upload")
def asr_upload(request: Request, file: UploadFile = File(...)):
    task_id = str(uuid4())

    # 1) 保存上传文件
    filename = f"{task_id}_{file.filename}"
    save_path = os.path.join(UPLOAD_DIR, filename)
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # 2) 计算外部 base_url（优先使用反代头，Runpod 常用）
    xfh = request.headers.get("x-forwarded-host")
    xfp = request.headers.get("x-forwarded-proto", "https")
    if xfh:
        base_url = f"{xfp}://{xfh}"
    else:
        base_url = str(request.base_url).rstrip("/")

    # 3) 初始化任务（放锁内）
    with lock:
        tasks[task_id] = {
            "status": "queued",
            "progress": 5,
            "audio_path": save_path,
            # 以后统一把这个字段当 **字符串** 使用
            "request": base_url,
            "context_prompt": None,
            "result": None,
            "logs": [f"upload_ok: path={save_path}, size={os.path.getsize(save_path)}B"],
        }

    # 4) 起后台线程
    threading.Thread(target=_process_task, args=(task_id,), daemon=True).start()

    # 5) 返回
    return {"task_id": task_id, "status": "queued"}



@app.post("/asr/start")
def asr_start(request: Request, body: StartBody):
    if not body.audio_url: raise HTTPException(400, "audio_url 必填")
    task_id = str(uuid.uuid4())
    ext = os.path.splitext(body.audio_url.split("?")[0])[-1] or ".wav"
    path = os.path.join(UPLOAD_DIR, f"{task_id}{ext}")
    r = requests.get(body.audio_url, timeout=300)
    if r.status_code != 200:
        raise HTTPException(r.status_code, f"下载失败：{r.text[:200]}")
    with open(path, "wb") as f: f.write(r.content)
    with lock:
        tasks[task_id] = {"status":"queued","progress":0,"request":request, "audio_path":path,
                          "series_id": body.series_id or "", "meeting_id": body.meeting_id or "",
                          "context_prompt": body.context_prompt or "", "glossary": body.glossary or []}
    threading.Thread(target=_process_task, args=(task_id,), daemon=True).start()
    return {"task_id": task_id, "status": "queued"}

@app.get("/asr/status")
def asr_status(task_id: str):
    with lock:
        t = tasks.get(task_id)
        if not t:
            return {"task_id": task_id, "status": "not_found"}
        return {
            "task_id": task_id,
            "status": t["status"],
            "progress": t.get("progress", 0),
            "logs": t.get("logs", []),
        }
    
@app.get("/asr/result")
def asr_result(task_id: str):
    with lock:
        t = tasks.get(task_id)
        if not t: raise HTTPException(404, "task not found")
        if t["status"] != "succeeded": return {"task_id": task_id, "status": t["status"]}
        return t["result"]

def _process_task(task_id: str):
    # ---------- 0) 取任务&关键字段 ----------
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
        # 极端：tasks 取不到
        return

    # ---------- 1) 模拟分支 ----------
    if SIMULATE:
        try:
            time.sleep(SIM_DELAY)
            demo = [
                {"start": 0, "end": 3, "text": "模拟-欢迎", "speaker": "SPK_00"},
                {"start": 3, "end": 7, "text": "模拟-讨论", "speaker": "SPK_00"},
                {"start": 7, "end": 11, "text": "模拟-总结", "speaker": "SPK_01"},
            ]
            srt_path  = os.path.join(RESULT_DIR, f"{task_id}.srt")
            json_path = os.path.join(RESULT_DIR, f"{task_id}.json")
            os.makedirs(RESULT_DIR, exist_ok=True)
            _write_srt(demo, srt_path)
            _save_json({"task_id": task_id, "segments": demo}, json_path)
            preview = "\n".join(
                ((f"[{d.get('speaker')}] " if d.get('speaker') else "") + d["text"]).strip()
                for d in demo
            ).strip()
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
        except Exception as e:
            import traceback
            tb = traceback.format_exc(limit=2)
            with lock:
                t = tasks.get(task_id) or {}
                _fail_task(t, f"{type(e).__name__}: {e}")
                _append_log(t, tb)
        return  # 模拟分支结束

    # ---------- 2) 实际处理分支 ----------
    try:
        # 2.1 预处理音频：统一成 16k/mono，避免 m4a 早期失败
        with lock:
            t = tasks[task_id]
            _append_log(t, "stage=prepare_audio")
            t["progress"] = 10
        norm_audio = to_wav16k_mono(audio_path)
        with lock:
            t = tasks[task_id]
            _append_log(t, f"audio_prepared: {norm_audio}")
            t["progress"] = 15

        # 2.2 ASR（faster-whisper）
        with lock:
            t = tasks[task_id]
            _append_log(t, "stage=asr_start")
            t["progress"] = 25
        model = _asr_load_model()  # 你已有的加载函数
        segments, info = model.transcribe(
            norm_audio,
            vad_filter=True,
            word_timestamps=True,
            initial_prompt=ctx_prompt,
            language=None,
            condition_on_previous_text=False,
            temperature=0.0,
        )
        words: List[Dict[str, Any]] = []
        for seg in segments:
            if seg.words:
                for w in seg.words:
                    words.append({"start": float(w.start), "end": float(w.end), "text": w.word})
        with lock:
            t = tasks[task_id]
            _append_log(t, f"stage=asr_done: words={len(words)}")
            t["progress"] = 55

        # 2.3 Diarization（失败不致命）
        diar_turns: List[Tuple[float, float, str]] = []
        enable_diar = (DIARIZE == "on") or (DIARIZE == "auto" and PYANNOTE_TOKEN)
        with lock:
            t = tasks[task_id]
            _append_log(t, f"diarization_enable={enable_diar} (DIARIZE={DIARIZE}, token={'yes' if PYANNOTE_TOKEN else 'no'})")
        try:
            if enable_diar:
                from pyannote.audio import Pipeline
                pipeline = Pipeline.from_pretrained(
                    "pyannote/speaker-diarization-3.1",
                    use_auth_token=PYANNOTE_TOKEN or None
                )
                # 确保 _resolve_device 返回 torch.device（前面已修过）
                pipeline.to(_device_for_pyannote())
                # 给 pyannote 用规范化后的 wav（更稳）
                wav_for_diar = _safe_wav_for_diar(norm_audio)
                diar = pipeline(wav_for_diar)
                for turn, _, speaker in diar.itertracks(yield_label=True):
                    diar_turns.append((float(turn.start), float(turn.end), str(speaker)))
                with lock:
                    t = tasks[task_id]
                    _append_log(t, f"diarization_ok: turns={len(diar_turns)}")
                    t["progress"] = 75
            else:
                with lock:
                    t = tasks[task_id]
                    _append_log(t, "diarization_skipped")
        except Exception as e:
            import traceback
            tb = traceback.format_exc(limit=1)
            with lock:
                t = tasks[task_id]
                _append_log(t, f"diarization_failed: {type(e).__name__}: {e}")
                _append_log(t, tb)
            diar_turns = []  # 不中断

        # ★ 2.4 逐词贴 speaker → 合并（遇 speaker 变化就切段）→ 重命名 → 清洗
        with lock:
            t = tasks[task_id]
            _append_log(t, "stage=align_start")

        # 给每个词打上 speaker（若 diar 失败则全为 None）
        words = assign_speaker_to_words(words, diar_turns)

        # 合并（确保 finalize_segments 尊重 word['speaker'] 边界）
        final_segments = finalize_segments(words, diar_turns, lang_hint="zh")
        
        # 让第一个开口的人从 SPEAKER_00 开始
        final_segments = renumber_speakers_by_first_appearance(final_segments)


        # 文本清洗：去零宽 + 你已有的 clean_text
        for seg in final_segments:
            seg["text"] = _strip_invisibles(seg["text"])
            seg["text"] = clean_text(seg["text"], lang_hint="auto")

        # 过滤空段（可选）
        final_segments = [s for s in final_segments if s["text"]]



        # 2.5 写出 SRT/JSON
        srt_path  = os.path.join(RESULT_DIR, f"{task_id}.srt")
        json_path = os.path.join(RESULT_DIR, f"{task_id}.json")
        os.makedirs(RESULT_DIR, exist_ok=True)
        _write_srt(final_segments, srt_path)
        _save_json({"task_id": task_id, "segments": final_segments}, json_path)

        preview = "\n".join(
            ((f"[{s.get('speaker')}] " if s.get("speaker") else "") + s["text"]).strip()
            for s in final_segments[:12]
        ).strip()

        with lock:
            t = tasks[task_id]
            t["result"] = {
                "aligned_preview": preview,
                "srt_url": _public_url(req, srt_path),
                "json_url": _public_url(req, json_path),
                "segments": final_segments,          # ← 新增：直接带回分段
            }
            t["status"] = "succeeded"
            t["progress"] = 100
            _append_log(t, "done")

    except Exception as e:
        import traceback
        tb = traceback.format_exc(limit=3)
        with lock:
            t = tasks.get(task_id) or {}
            _fail_task(t, f"{type(e).__name__}: {e}")
            _append_log(t, tb)