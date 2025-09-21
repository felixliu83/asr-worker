# align.py
# -*- coding: utf-8 -*-

from __future__ import annotations
from typing import List, Dict, Any, Tuple, Optional
import re

__all__ = [
    "assign_speaker_to_words",
    "merge_words_to_segments",
    "renumber_speakers_by_first_appearance",
    "attach_speaker_by_overlap",
    "finalize_segments",
]

# 句末/强分割标点（中文+英文）
_PUNCT_BREAK = re.compile(r"[。！？!?]|(\.\s)|[.…]+")

# ---- 小工具 -----------------------------------------------------------------

def _smart_join(prev: str, token: str) -> str:
    """尽量在英文字母/数字之间加空格，其它语言直接拼接。"""
    if not prev:
        return token
    # 前一字符和后一 token 的首字符都为 ASCII 时，补空格
    if prev[-1].isascii() and token[:1].isascii():
        return prev + " " + token
    return prev + token


# ---- 逐词贴 speaker ---------------------------------------------------------

def assign_speaker_to_words(words: List[Dict[str, Any]],
                            turns: List[Tuple[float, float, str]]) -> List[Dict[str, Any]]:
    """
    给每个词贴上说话人。
    words: [{"start": float, "end": float, "text": str}, ...]
    turns: [(ts, te, "SPEAKER_xx"), ...]  —— pyannote 输出的 (时间区间, 说话人)
    规则：
      1) 用词的中点命中 turn 则取该 speaker；
      2) 若未命中，取“最近的 turn”作为兜底。
    """
    if not turns:
        for w in words:
            w["speaker"] = None
        return words

    for w in words:
        ws, we = float(w["start"]), float(w["end"])
        mid = 0.5 * (ws + we)

        spk: Optional[str] = None
        # 命中包含区间
        for (ts, te, s) in turns:
            if ts <= mid <= te:
                spk = s
                break

        # 最近距离兜底
        if spk is None:
            best = (1e9, None)  # (distance, speaker)
            for (ts, te, s) in turns:
                d = 0.0 if ts <= mid <= te else min(abs(mid - ts), abs(mid - te))
                if d < best[0]:
                    best = (d, s)
            spk = best[1]

        w["speaker"] = spk
    return words


# ---- 词 → 句段：换人=硬边界 ---------------------------------------------------

def merge_words_to_segments(words: List[Dict[str, Any]],
                            *,
                            max_gap: float = 0.60,
                            max_dur: float = 8.0) -> List[Dict[str, Any]]:
    """
    用“说话人变化=硬分割”合并词为句段，同时结合停顿与最长时长控制。
    参数：
      max_gap: 两个词间隔超过该阈值(秒)则切段
      max_dur: 单段最长持续时间(秒)，超出则强制切
    说明：
      - 要求 words 已包含 w['speaker']（使用 assign_speaker_to_words 先行赋值）
      - 标点只做“软切”倾向；真正切不切由换人/间隔/时长控制
    """
    if not words:
        return []

    segs: List[Dict[str, Any]] = []
    cur: Optional[Dict[str, Any]] = None
    last_end: Optional[float] = None

    def flush():
        nonlocal cur
        if cur and cur["text"].strip():
            cur["text"] = cur["text"].strip()
            segs.append(cur)
        cur = None

    for w in words:
        ws, we = float(w["start"]), float(w["end"])
        spk = w.get("speaker")
        txt = w["text"]

        if cur is None:
            cur = {"start": ws, "end": we, "speaker": spk, "text": txt}
            last_end = we
            continue

        # 1) 说话人改变：硬切
        if spk != cur.get("speaker"):
            flush()
            cur = {"start": ws, "end": we, "speaker": spk, "text": txt}
            last_end = we
            continue

        # 2) 间隔过大：切
        gap = ws - (last_end if last_end is not None else ws)
        if gap > max_gap:
            flush()
            cur = {"start": ws, "end": we, "speaker": spk, "text": txt}
            last_end = we
            continue

        # 3) 追加
        cur["end"] = we
        cur["text"] = _smart_join(cur["text"], txt)

        # 4) 句末标点：软切（段已>=1.5s 时更倾向切）
        if _PUNCT_BREAK.search(txt) and (cur["end"] - cur["start"]) >= 1.5:
            flush()

        # 5) 时长过长：强切
        if cur and (cur["end"] - cur["start"]) >= max_dur:
            flush()

        last_end = we

    flush()
    return segs


# ---- 首现顺序重命名 -----------------------------------------------------------

def renumber_speakers_by_first_appearance(segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    将说话人按首次出现顺序重命名为 SPEAKER_00, SPEAKER_01, ...
    """
    mapping: Dict[str, str] = {}
    nxt = 0
    for seg in segments:
        spk = seg.get("speaker")
        if spk is None:
            continue
        if spk not in mapping:
            mapping[spk] = f"SPEAKER_{nxt:02d}"
            nxt += 1
        seg["speaker"] = mapping[spk]
    return segments


# ---- 兼容：整句 IoU 贴 speaker（不建议再用，但保留以兼容旧代码） ----------------

def attach_speaker_by_overlap(units: List[Dict[str, Any]],
                              turns: List[Tuple[float, float, str]]) -> List[Dict[str, Any]]:
    """
    对“整句”按 IoU 贴 speaker 的旧实现（不建议——会把跨 speaker 的句子贴成一个人）。
    仅保留以兼容旧代码。
    """
    if not turns:
        return [{**u, "speaker": None} for u in units]
    out = []
    last_spk: Optional[str] = None
    for u in units:
        us, ue = float(u["start"]), float(u["end"])
        best_iou = 0.0
        best_spk: Optional[str] = None
        for (ts, te, spk) in turns:
            inter = max(0.0, min(ue, te) - max(us, ts))
            union = max(ue, te) - min(us, ts)
            iou = (inter / union) if union > 0 else 0.0
            if inter > 0.0 and iou > best_iou:
                best_iou = iou
                best_spk = spk
        spk_to_use = best_spk if best_spk is not None else last_spk
        out_unit = {**u, "speaker": spk_to_use}
        out.append(out_unit)
        if spk_to_use is not None:
            last_spk = spk_to_use
    return out


# ---- （可选）同说话人短段再合并 ------------------------------------------------

def merge_short_segments_same_speaker(segments: List[Dict[str, Any]],
                                      min_dur: float = 0.30) -> List[Dict[str, Any]]:
    """
    仅当相邻两段的 speaker 相同，且后一段很短（<min_dur），则与前段合并。
    防止逐词切段后过碎。
    """
    out: List[Dict[str, Any]] = []
    for seg in segments:
        if not out:
            out.append(seg)
            continue
        dur = float(seg["end"]) - float(seg["start"])
        if dur < min_dur and seg.get("speaker") == out[-1].get("speaker"):
            out[-1]["end"] = seg["end"]
            out[-1]["text"] = _smart_join(out[-1]["text"], seg["text"])
        else:
            out.append(seg)
    return out


# ---- 总入口：finalize --------------------------------------------------------

def finalize_segments(words: List[Dict[str, Any]],
                      diar_turns: List[Tuple[float, float, str]],
                      lang_hint: str = "auto") -> List[Dict[str, Any]]:
    """
    最终对齐流程：
      1) 逐词贴 speaker（命中/最近兜底）
      2) 以“说话人变化=硬边界”合并成句段（再结合停顿/时长）
      3) （可选）同说话人短段再合并，减少碎片
      4) 按首现顺序重命名为 SPEAKER_00/01/...
    注意：不再使用“整句 IoU 贴 speaker”的旧路径，避免把多说话人塞进一条句子。
    """
    # 1) 逐词贴 speaker
    words = assign_speaker_to_words(words, diar_turns)

    # 2) 合并为句段（换人=切）
    sents = merge_words_to_segments(words, max_gap=0.60, max_dur=8.0)

    # 3) （可选）同 speaker 的短段合并（阈值可按需调整/关闭）
    sents = merge_short_segments_same_speaker(sents, min_dur=0.30)

    # 4) 首现顺序重命名
    sents = renumber_speakers_by_first_appearance(sents)

    return sents