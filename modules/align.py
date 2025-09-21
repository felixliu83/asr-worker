# modules/align.py
import re
from typing import List, Dict, Tuple, Any

_CJK   = r"[\u4e00-\u9fff]"
_OPEN  = "（【《「『(\"“‘[({"
_CLOSE = "，。？！；：、）】》」』)\"”’]}\'.,"  # 加入中英标点与引号
_PUNCT = "。？！.!?"                         # 句末标点：中英都考虑

def smart_join(prev: str, token: str) -> str:
    """智能拼接：中文-中文、标点前、开括号后、英文缩写的 `'` 都不加空格；其他情况按需加空格。"""
    if not prev:
        return token
    if token and (token[0] in _CLOSE or token.startswith("'")):  # 标点或缩写
        return prev + token
    if prev and prev[-1] in _OPEN:                               # 开括号/引号后
        return prev + token
    if re.match(_CJK, prev[-1]) and re.match(_CJK, token[0]):    # 中-中
        return prev + token
    # 默认：加一个空格（避免重复空格）
    return prev + (" " if not prev.endswith(" ") else "") + token

def merge_short_segments(segments: List[Dict[str, Any]], min_dur: float=1.2, max_gap: float=0.6) -> List[Dict[str,Any]]:
    if not segments: return []
    out = []
    cur = segments[0].copy()
    for seg in segments[1:]:
        gap = seg["start"] - cur["end"]
        cur_dur = cur["end"] - cur["start"]
        if cur_dur < min_dur or gap <= max_gap:
            cur["end"] = seg["end"]
            cur["text"] = smart_join(cur["text"].rstrip(), seg["text"].lstrip())
            if cur.get("speaker") != seg.get("speaker"):
                cur["speaker"] = None
        else:
            out.append(cur); cur = seg.copy()
    out.append(cur)
    return out

def cut_sentences_by_punct(words: List[Dict[str,Any]], punct: str=_PUNCT, gap: float=0.8) -> List[Dict[str,Any]]:
    """用标点边界与时间间隔把词级合并为句段；兼容英文 '.' '!' '?'。"""
    if not words: return []
    sents = []
    cur = {"start": words[0]["start"], "end": words[0]["end"], "text": words[0]["text"]}
    for w in words[1:]:
        boundary = (w["text"] in punct) or (w["start"] - cur["end"] > gap)
        if boundary:
            sents.append(cur)
            cur = {"start": w["start"], "end": w["end"], "text": w["text"]}
        else:
            cur["end"] = w["end"]
            cur["text"] = smart_join(cur["text"], w["text"])
    sents.append(cur)
    # 尾部孤立标点并入上一句
    out = []
    for s in sents:
        token = s["text"].strip()
        if len(token) == 1 and token in punct:
            if out:
                out[-1]["end"] = s["end"]
                out[-1]["text"] = smart_join(out[-1]["text"], token)
        else:
            out.append(s)
    return out

def attach_speaker_by_overlap(units: List[Dict[str, Any]],
                              turns: List[Tuple[float, float, str]]) -> List[Dict[str, Any]]:
    """
    按 IoU 最大给句段贴说话人；若当前句段与任何 turn 无重叠，则沿用上一个说话人（保底策略）。
    units: [{"start": float, "end": float, "text": str, ...}, ...]
    turns: [(start: float, end: float, speaker: str), ...]
    """
    if not turns:
        return [{**u, "speaker": None} for u in units]

    out: List[Dict[str, Any]] = []
    last_spk: Optional[str] = None

    for u in units:
        us, ue = float(u["start"]), float(u["end"])
        best_iou = 0.0
        best_spk: Optional[str] = None

        for (ts, te, spk) in turns:
            # 交并比（IoU）
            inter = max(0.0, min(ue, te) - max(us, ts))
            union = max(ue, te) - min(us, ts)
            iou = (inter / union) if union > 0 else 0.0

            if inter > 0.0 and iou > best_iou:
                best_iou = iou
                best_spk = spk

        # 保底：若没有任何重叠（best_spk 仍为 None），沿用上一个说话人
        spk_to_use = best_spk if best_spk is not None else last_spk

        out_unit = {**u, "speaker": spk_to_use}
        out.append(out_unit)

        # 只有在当前得到明确说话人时，才更新 last_spk
        if spk_to_use is not None:
            last_spk = spk_to_use

    return out

def finalize_segments(words: List[Dict[str,Any]],
                      diar_turns: List[Tuple[float,float,str]],
                      lang_hint: str="zh") -> List[Dict[str,Any]]:
    sents = cut_sentences_by_punct(words, punct=_PUNCT, gap=0.8)
    sents = attach_speaker_by_overlap(sents, diar_turns)
    sents = merge_short_segments(sents, min_dur=1.2, max_gap=0.6)
    return sents