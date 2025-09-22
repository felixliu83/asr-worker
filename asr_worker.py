import os, json, uuid, time, threading, torch, requests, subprocess, shutil, tempfile, os, re
from typing import Dict, Any, Optional, List, Tuple
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import numpy as np
from uuid import uuid4
from modules.text_clean import clean_text
from modules.align import finalize_segments, assign_speaker_to_words, renumber_speakers_by_first_appearance
from faster_whisper import WhisperModel
from typing import Optional


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

ZW_RE = re.compile(r"[\x00-\x08\x0B-\x0C\x0E-\x1F\u200B-\u200F\u202A-\u202E\u2060-\u206F\uFEFF]")

INVIS_RE = re.compile(r"[\x00-\x08\x0B-\x0C\x0E-\x1F\u200B-\u200F\u202A-\u202E\u2060-\u206F\uFEFF]")
def _strip_invisibles(s: str) -> str:
    return INVIS_RE.sub("", s or "")


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
        "device": str(_resolve_device()),
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
    """
    新版处理流程（最小侵入）：
      1) 统一音频 -> norm_audio
      2) WhisperX: 文本 + 高精度词级时间戳
      3) Pyannote: 说话人 -> 10ms 帧级标签（含轻量平滑）
      4) 词级多数派赋 speaker
      5) finalize_segments（你 modules/align.py 的覆盖版）
      6) 强清洗 -> SRT/JSON/预览 -> 返回
    """
    import os, time, json, re, numpy as np
    from typing import List, Dict, Any, Tuple
    import threading

    # ---- 工具 ----
    def _append(t, msg): 
        try:
            _append_log(t, msg)
        except Exception:
            pass

    # 强清洗（零宽/控制符）
    INVIS_RE = re.compile(r"[\x00-\x08\x0B-\x0C\x0E-\x1F\u200B-\u200F\u202A-\u202E\u2060-\u206F\uFEFF]")
    def _strip_invisibles(s: str) -> str:
        return INVIS_RE.sub("", s or "")

    # A. WhisperX：ASR + 对齐，取高精度词级时间戳
    def transcribe_with_whisperx(
        audio_path: str,
        model_name: str,
        device: str,
        compute_type: str = "float16",
        lang_hint: Optional[str] = None,
        ctx_prompt: Optional[str] = None,  # 注意：whisperx 不支持 initial_prompt，这个参数仅占位
    ):
        """
        用 whisperx 做转写 + 对齐，返回 (words, asr_raw)
        - whisperx 的 FasterWhisperPipeline.transcribe 不支持 initial_prompt
        - lang_hint: "zh" / "en" / None ；None 时让 whisperx 自动检测
        """
        import whisperx
        import torch

        # 1) 加载 ASR 模型
        # whisperx 3.3.1：compute_type 建议 "float16" (CUDA) / "int8" (CPU)
        if device == "cuda":
            ct = "float16" if "float16" in compute_type else "int8_float16"
        else:
            ct = "int8"

        model = whisperx.load_model(model_name, device, compute_type=ct)

        # 2) 转写（无 initial_prompt）
        # 仅传入 whisperx 支持的参数：language / task / batch_size / vad / chunk_size / print_progress
        asr = model.transcribe(
            audio_path,
            language=(None if (not lang_hint or lang_hint == "auto") else lang_hint),
            task="transcribe",
            batch_size=8,
            vad=True,
            print_progress=False,
        )
        # asr 结构里含 'segments' 和 'language'
        detected_lang = asr.get("language") or (lang_hint if lang_hint and lang_hint != "auto" else None)

        # 3) 词级对齐（whisperx 内置）
        # 语言代码给 align_model 用：whisperx 需要两位/ISO 代码，如 'zh'、'en'
        am, meta = whisperx.load_align_model(
            language_code=(detected_lang or "zh"),  # 兜底 zh，避免有些短音频未返 language
            device=device,
        )
        aligned = whisperx.align(asr["segments"], am, meta, audio_path, device)

        # 4) 输出 words（兼容你后面的 finalize 流程）
        words = []
        for seg in aligned.get("segments", []):
            for w in seg.get("words", []) or []:
                # 过滤无时间戳或空 token
                if w.get("word") and (w.get("start") is not None) and (w.get("end") is not None):
                    words.append({
                        "start": float(w["start"]),
                        "end": float(w["end"]),
                        "text": w["word"],
                    })

        # 5) 返回 “原始 asr+对齐结果”（便于调试/保存）
        asr_raw = {
            "language": detected_lang,
            "segments": aligned.get("segments", asr.get("segments", [])),
        }

        return words, asr_raw

    # B. Diarization：转为 10ms 帧级说话人标签 + 轻量平滑
    def diarize_to_frames(audio_path, device="cuda", num_speakers=None, min_spk=2, max_spk=5, frame_hop=0.01):
        from pyannote.audio import Pipeline
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=os.environ.get("PYANNOTE_TOKEN", None)
        )
        if num_speakers is not None:
            diar = pipeline(audio_path, num_speakers=num_speakers)
        else:
            diar = pipeline(audio_path, min_speakers=min_spk, max_speakers=max_spk)

        # 全时长
        total_end = 0.0
        for turn, _, _ in diar.itertracks(yield_label=True):
            total_end = max(total_end, float(turn.end))
        T = int(np.ceil(total_end / frame_hop))
        frames = [None] * T

        # 简化赋值：后写覆盖；需要更严谨可统计每帧各 spk 时长再 argmax
        for turn, _, spk in diar.itertracks(yield_label=True):
            s = int(np.floor(float(turn.start) / frame_hop))
            e = int(np.ceil(float(turn.end)   / frame_hop))
            for t in range(max(0, s), min(T, e)):
                frames[t] = spk

        # 轻量平滑：吸收 <120ms 的瞬时切换（连续性惩罚）
        smooth = []
        last = None
        run = 0
        min_run = int(0.12 / frame_hop)
        for t in range(T):
            cur = frames[t]
            if cur == last:
                run += 1
            else:
                if run > 0 and run < min_run and len(smooth) > run:
                    for k in range(run):
                        smooth[-1 - k] = smooth[-1 - run]  # 回填为上上个说话人
                last = cur
                run = 1
            smooth.append(last)
        return smooth, frame_hop

    # C. 词级多数派赋 speaker（用帧标签投票）
    def assign_speaker_by_frames(words: List[Dict[str, Any]], frame_labels, hop=0.01):
        out = []
        nF = len(frame_labels)
        for w in words:
            ws, we = float(w["start"]), float(w["end"])
            s = max(0, int(ws / hop))
            e = max(s, int(np.ceil(we / hop)))
            votes = {}
            for t in range(s, min(e, nF)):
                spk = frame_labels[t]
                if spk is None:
                    continue
                votes[spk] = votes.get(spk, 0) + 1
            spk = max(votes.items(), key=lambda x: x[1])[0] if votes else None
            out.append({**w, "speaker": spk})
        return out

    # ---------- 0) 取任务 ----------
    try:
        with lock:
            t = tasks.get(task_id)
            if not t:
                return
            t["status"] = "processing"; t["progress"] = 5
            audio_path = t.get("audio_path")
            req = t.get("request")
            ctx_prompt = t.get("context_prompt") or None
            _append(t, "start_processing")
            if not audio_path or not os.path.exists(audio_path):
                _fail_task(t, f"audio_missing: {audio_path}")
                return
    except Exception:
        return

    # ---------- 1) 模拟分支 ----------
    if SIMULATE:
        try:
            time.sleep(SIM_DELAY)
            demo = [
                {"start": 0, "end": 3, "text": "模拟-欢迎", "speaker": "SPEAKER_00"},
                {"start": 3, "end": 7, "text": "模拟-讨论", "speaker": "SPEAKER_00"},
                {"start": 7, "end": 11, "text": "模拟-总结", "speaker": "SPEAKER_01"},
            ]
            srt_path  = os.path.join(RESULT_DIR, f"{task_id}.srt")
            json_path = os.path.join(RESULT_DIR, f"{task_id}.json")
            os.makedirs(RESULT_DIR, exist_ok=True)
            _write_srt(demo, srt_path)
            _save_json({"task_id": task_id, "segments": demo}, json_path)
            preview = "\n".join(((f"[{d.get('speaker')}] " if d.get('speaker') else "") + d["text"]).strip() for d in demo).strip()
            with lock:
                t = tasks[task_id]
                t["status"] = "succeeded"; t["progress"] = 100
                t["result"] = {
                    "aligned_preview": preview,
                    "srt_url": _public_url(req, srt_path),
                    "json_url": _public_url(req, json_path),
                    "segments": demo,
                }
                _append(t, "done (simulate)")
        except Exception as e:
            import traceback
            tb = traceback.format_exc(limit=2)
            with lock:
                t = tasks.get(task_id) or {}
                _fail_task(t, f"{type(e).__name__}: {e}")
                _append(t, tb)
        return

    # ---------- 2) 实际处理 ----------
    try:
        # 2.1 统一音频
        with lock:
            t = tasks[task_id]; t["progress"] = 10; _append(t, "stage=prepare_audio")
        norm_audio = to_wav16k_mono(audio_path)
        with lock:
            t = tasks[task_id]; t["progress"] = 15; _append(t, f"audio_prepared: {norm_audio}")

        lang_hint: Optional[str] = None
        try:
            # 若你在 /asr/upload 里把语言传进来了，可这样取：
            lang_hint = t.get("lang_hint") or t.get("request", {}).get("lang_hint")
        except Exception:
            pass
        if not lang_hint:
            lang_hint = os.getenv("LANG_HINT", "auto")  # 'zh' / 'en' / 'auto'


        # 2.2 WhisperX 对齐（高精度词时间戳）
        with lock:
            t = tasks[task_id]; t["progress"] = 25; _append(t, "stage=asr_whisperx_start")
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        words, asr_raw = transcribe_with_whisperx(
            norm_audio, WHISPER_MODEL, device, compute_type=COMPUTE_TYPE,
            lang_hint=lang_hint, ctx_prompt=ctx_prompt
        )
        with lock:
            t = tasks[task_id]; t["progress"] = 40; _append_log(tasks[task_id], f"asr_lang={asr_raw.get('language') or lang_hint}")

        # 2.3 Diarization -> 10ms 帧（人数优先取环境变量）
        with lock:
            t = tasks[task_id]; _append(t, f"stage=diar_frames_start (token={'yes' if os.environ.get('PYANNOTE_TOKEN') else 'no'})")
        num_spk_env = os.environ.get("DIAR_NUM_SPEAKERS")
        num_spk = int(num_spk_env) if (num_spk_env or "").isdigit() else None
        try:
            frame_labels, hop = diarize_to_frames(
                norm_audio, device=device, num_speakers=num_spk,
                min_spk=2, max_spk=5, frame_hop=0.01
            )
            with lock:
                t = tasks[task_id]; t["progress"] = 60; _append(t, f"diar_frames_ok: T={len(frame_labels)} hop={hop}")
        except Exception as e:
            # diarization 失败不致命：全部 None
            frame_labels, hop = [], 0.01
            with lock:
                t = tasks[task_id]; _append(t, f"diar_frames_failed: {type(e).__name__}: {e}")

        # 2.4 词级多数派赋 speaker
        if frame_labels:
            words_spk = assign_speaker_by_frames(words, frame_labels, hop)
        else:
            # 无帧信息则保持 None（finalize 会尽力做）
            words_spk = [{**w, "speaker": None} for w in words]
        with lock:
            t = tasks[task_id]; t["progress"] = 70; _append(t, "word_speakers_assigned_by_frames")

        # 2.5 词->句 段落生成（使用你 modules/align.py 的 finalize_segments）
        from modules.align import finalize_segments
        final_segments = finalize_segments(words_spk, diar_turns=[], lang_hint="zh")
        with lock:
            t = tasks[task_id]; t["progress"] = 80; _append(t, f"finalize_done: segs={len(final_segments)}")

        # 2.6 强清洗（必须在写盘/返回前）
        from modules.text_clean import clean_text  # 你现有的清洗
        for seg in final_segments:
            seg["text"] = _strip_invisibles(seg["text"])
            seg["text"] = clean_text(seg["text"], lang_hint="auto")
        final_segments = [s for s in final_segments if s["text"]]

        # 2.7 写盘
        srt_path  = os.path.join(RESULT_DIR, f"{task_id}.srt")
        json_path = os.path.join(RESULT_DIR, f"{task_id}.json")
        os.makedirs(RESULT_DIR, exist_ok=True)
        _write_srt(final_segments, srt_path)
        _save_json({"task_id": task_id, "segments": final_segments}, json_path)

        # 2.8 预览 + 返回
        preview = "\n".join(
            ((f"[{s.get('speaker')}] " if s.get('speaker') else "") + s["text"]).strip()
            for s in final_segments[:12]
        ).strip()
        with lock:
            t = tasks[task_id]
            t["result"] = {
                "aligned_preview": preview,
                "srt_url": _public_url(req, srt_path),
                "json_url": _public_url(req, json_path),
                "segments": final_segments,
            }
            t["status"] = "succeeded"; t["progress"] = 100
            _append(t, "done")
    except Exception as e:
        import traceback
        tb = traceback.format_exc(limit=3)
        with lock:
            t = tasks.get(task_id) or {}
            _fail_task(t, f"{type(e).__name__}: {e}")
            _append(t, tb)