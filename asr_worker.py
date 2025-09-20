import os, json, uuid, time, threading
from typing import Dict, Any, Optional, List, Tuple
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import requests

from modules.text_clean import clean_text
from modules.align import finalize_segments

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

def _resolve_device():
    if DEVICE == "auto":
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"
    return DEVICE

def _asr_load_model():
    from faster_whisper import WhisperModel
    device = _resolve_device()
    compute = COMPUTE_TYPE
    if device == "cpu" and compute not in ("int8", "int8_float16", "int16"):
        compute = "int8"
    return WhisperModel(WHISPER_MODEL, device=device, compute_type=compute)

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

def _public_url(request: Optional[Request], filepath: str) -> str:
    base = str(request.base_url).rstrip("/") if request else ""
    return f"{base}/results/{os.path.basename(filepath)}"

class StartBody(BaseModel):
    audio_url: str
    series_id: Optional[str] = ""
    meeting_id: Optional[str] = ""
    context_prompt: Optional[str] = ""
    glossary: Optional[List[str]] = []

@app.get("/healthz")

def healthz():
    version = "unknown"
    try:
        with open("/workspace/VERSION") as f:
            version = f.read().strip()
    except Exception:
        pass
    return {
        "ok": True,
        "model": WHISPER_MODEL,
        "device": DEVICE,
        "simulate": SIMULATE,
        "version": version,
    }

@app.post("/asr/upload")
def asr_upload(request: Request, file: UploadFile = File(...)):
    ext = os.path.splitext(file.filename or "")[1] or ".wav"
    task_id = str(uuid.uuid4())
    path = os.path.join(UPLOAD_DIR, f"{task_id}{ext}")
    with open(path, "wb") as f: f.write(file.file.read())
    with lock:
        tasks[task_id] = {"status":"queued","progress":0,"request":request, "audio_path":path}
    threading.Thread(target=_process_task, args=(task_id,), daemon=True).start()
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
        if not t: return {"task_id": task_id, "status": "not_found"}
        return {"task_id": task_id, "status": t["status"], "progress": t.get("progress", 0)}

@app.get("/asr/result")
def asr_result(task_id: str):
    with lock:
        t = tasks.get(task_id)
        if not t: raise HTTPException(404, "task not found")
        if t["status"] != "succeeded": return {"task_id": task_id, "status": t["status"]}
        return t["result"]

def _process_task(task_id: str):
    try:
        with lock:
            t = tasks[task_id]
            t["status"] = "processing"; t["progress"] = 5
            audio_path = t["audio_path"]; req = t.get("request")
            ctx_prompt = t.get("context_prompt") or None

        if SIMULATE:
            time.sleep(SIM_DELAY)
            demo = [{"start":0,"end":3,"text":"模拟-欢迎","speaker":"SPK_00"},
                    {"start":3,"end":7,"text":"模拟-讨论","speaker":"SPK_00"},
                    {"start":7,"end":11,"text":"模拟-总结","speaker":"SPK_01"}]
            srt_path  = os.path.join(RESULT_DIR, f"{task_id}.srt")
            json_path = os.path.join(RESULT_DIR, f"{task_id}.json")
            _write_srt(demo, srt_path)
            _save_json({"task_id":task_id,"segments":demo}, json_path)
            with lock:
                tasks[task_id]["status"]="succeeded"; tasks[task_id]["progress"]=100
                tasks[task_id]["result"]={
                    "aligned_preview":"\n".join(((f"[{d['speaker']}] " if d.get('speaker') else "")+d['text']).strip() for d in demo),
                    "srt_url": _public_url(req, srt_path), "json_url": _public_url(req, json_path)
                }
            return

        # 1) ASR
        from faster_whisper import WhisperModel
        model = _asr_load_model()
        with lock: tasks[task_id]["progress"] = 10
        segments, info = model.transcribe(
            audio_path,
            vad_filter=True,
            word_timestamps=True,
            initial_prompt=ctx_prompt,
            condition_on_previous_text=True,
            temperature=0.0,
        )

        words: List[Dict[str,Any]] = []
        for seg in segments:
            if seg.words:
                for w in seg.words:
                    words.append({"start": float(w.start), "end": float(w.end), "text": w.word})

        with lock: tasks[task_id]["progress"] = 55

        # 2) Diarization（可选）
        diar_turns: List[Tuple[float,float,str]] = []
        enable_diar = (DIARIZE == "on") or (DIARIZE == "auto" and PYANNOTE_TOKEN)
        if enable_diar:
            try:
                from pyannote.audio import Pipeline
                pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=PYANNOTE_TOKEN)
                pipeline.to(_resolve_device())
                diar = pipeline(audio_path)
                for turn, _, speaker in diar.itertracks(yield_label=True):
                    diar_turns.append((float(turn.start), float(turn.end), str(speaker)))
                with lock: tasks[task_id]["progress"] = 75
            except Exception as e:
                diar_turns = []
                with lock: tasks[task_id].setdefault("logs", []).append(f"diarization_failed: {e}")

        # 3) 对齐 + 清洗
        final_segments = finalize_segments(words, diar_turns, lang_hint="zh")
        for seg in final_segments:
            seg["text"] = clean_text(seg["text"], lang_hint="auto")
        final_segments = [s for s in final_segments if s["text"]]

        # 4) 保存产物
        srt_path  = os.path.join(RESULT_DIR, f"{task_id}.srt")
        json_path = os.path.join(RESULT_DIR, f"{task_id}.json")
        _write_srt(final_segments, srt_path)
        _save_json({"task_id": task_id, "segments": final_segments}, json_path)

        preview = "\n".join(((f"[{s.get('speaker')}] " if s.get('speaker') else "") + s["text"]).strip() for s in final_segments[:12])
        with lock:
            tasks[task_id]["status"]="succeeded"; tasks[task_id]["progress"]=100
            tasks[task_id]["result"] = {"aligned_preview": preview, "srt_url": _public_url(req, srt_path), "json_url": _public_url(req, json_path)}

    except Exception as e:
        with lock:
            tasks[task_id]["status"]="failed"; tasks[task_id]["error"] = str(e)
