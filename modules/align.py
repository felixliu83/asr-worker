# modules/align.py
# -*- coding: utf-8 -*-

from __future__ import annotations
from typing import List, Dict, Any, Tuple, Optional
import re
import string

__all__ = [
    "merge_ascii_fragments",
    "assign_speaker_to_words",
    "merge_words_to_segments",
    "merge_short_segments_same_speaker",
    "renumber_speakers_by_first_appearance",
    "attach_speaker_by_overlap",
    "finalize_segments",
]

# 句末/强分割标点（中文+英文）
_PUNCT_BREAK = re.compile(r"[。！？!?]|(\.\s)|[.…]+")


# -------------------- 文本拼接小工具 --------------------

def _smart_join(prev: str, token: str) -> str:
    """尽量在 ASCII 词之间补空格；中英文混排时减少粘连。"""
    if not prev:
        return token
    a = prev[-1]
    b = token[:1]
    if a.isascii() and b.isascii():
        return prev + " " + token
    # 前 ASCII、后中文：无需额外空格；中文标点直接拼接
    return prev + token


# -------------------- 合并英文碎片（I sa ac -> Isaac） --------------------

def _is_ascii_alpha(s: str) -> bool:
    return bool(s) and all(ch in string.ascii_letters for ch in s)

def merge_ascii_fragments(words: List[Dict[str, Any]],
                          max_gap: float = 0.20,
                          max_pieces: int = 3) -> List[Dict[str, Any]]:
    """
    将像 ["I","sa","ac"] 这类 ASCII 字母碎片在短时间内合并。
    规则：
      - 首片：单个大写字母（如 I）
      - 后续片：全小写，长度<=3，且与前片时间间隔 <= max_gap
      - 最多合并 max_pieces 个后续片
    合并后：
      - 文本：首字母 + 其余小写，如 "Isaac"
      - 时间：覆盖所有片段
    """
    if not words:
        return words

    out: List[Dict[str, Any]] = []
    i, n = 0, len(words)
    while i < n:
        w = words[i]
        txt = w["text"]
        ws, we = float(w["start"]), float(w["end"])

        if len(txt) == 1 and txt.isupper() and _is_ascii_alpha(txt):
            pieces = [txt]
            start, end = ws, we
            j = i + 1
            while j < n and (len(pieces) - 1) < max_pieces:
                nxt = words[j]
                gap = float(nxt["start"]) - end
                t = nxt["text"]
                if gap <= max_gap and _is_ascii_alpha(t) and t.islower() and 1 <= len(t) <= 3:
                    pieces.append(t)
                    end = float(nxt["end"])
                    j += 1
                else:
                    break
            if len(pieces) >= 2:
                merged = pieces[0] + "".join(pieces[1:])
                merged = merged[0] + merged[1:].lower()
                out.append({"start": start, "end": end, "text": merged})
                i = j
                continue

        out.append(w)
        i += 1

    return out


# -------------------- 逐词贴 speaker（最大重叠优先） --------------------

def assign_speaker_to_words(words: List[Dict[str, Any]],
                            turns: List[Tuple[float, float, str]]) -> List[Dict[str, Any]]:
    """
    用“与某个 turn 的重叠时长最大”来决定每个词的说话人；
    若无重叠，再用“最近端点”兜底。
    这样能显著减少边界处尾字/单词贴错人的情况。
    """
    if not turns:
        for w in words:
            w["speaker"] = None
        return words

    for w in words:
        ws, we = float(w["start"]), float(w["end"])

        # 先看与哪个 turn 重叠最多
        best_overlap = 0.0
        best_spk: Optional[str] = None
        for (ts, te, s) in turns:
            inter = max(0.0, min(we, te) - max(ws, ts))
            if inter > best_overlap:
                best_overlap = inter
                best_spk = s

        if best_spk is None:
            # 没重叠：用最近端点兜底（比 midpoint 更稳）
            mid = 0.5 * (ws + we)
            best = (1e9, None)
            for (ts, te, s) in turns:
                d = 0.0 if ts <= mid <= te else min(abs(mid - ts), abs(mid - te))
                if d < best[0]:
                    best = (d, s)
            best_spk = best[1]

        w["speaker"] = best_spk

    return words


# -------------------- 词 -> 段：换人=硬切 --------------------

def merge_words_to_segments(words: List[Dict[str, Any]],
                            *,
                            max_gap: float = 0.60,
                            max_dur: float = 8.0) -> List[Dict[str, Any]]:
    """
    基于词序列合并为句段。
    规则优先级：
      1) 说话人变化 -> 立刻切段（硬边界）
      2) 词间间隔 > max_gap -> 切段
      3) 命中句末标点（软切，段>=1.5s 时更倾向切）
      4) 段长 > max_dur -> 强制切
    要求：words 已经拥有 w['speaker']（先调用 assign_speaker_to_words）
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

        # 1) 换人 -> 硬切
        if spk != cur.get("speaker"):
            flush()
            cur = {"start": ws, "end": we, "speaker": spk, "text": txt}
            last_end = we
            continue

        # 2) 间隔太大 -> 切
        gap = ws - (last_end if last_end is not None else ws)
        if gap > max_gap:
            flush()
            cur = {"start": ws, "end": we, "speaker": spk, "text": txt}
            last_end = we
            continue

        # 3) 追加
        cur["end"] = we
        cur["text"] = _smart_join(cur["text"], txt)

        # 4) 句末标点 -> 软切（段已>=1.5s）
        if _PUNCT_BREAK.search(txt) and (cur["end"] - cur["start"]) >= 1.5:
            flush()

        # 5) 段过长 -> 强切
        if cur and (cur["end"] - cur["start"]) >= max_dur:
            flush()

        last_end = we

    flush()
    return segs


# -------------------- 同 speaker 短段再合并（可选） --------------------

def merge_short_segments_same_speaker(segments: List[Dict[str, Any]],
                                      min_dur: float = 0.30) -> List[Dict[str, Any]]:
    """
    仅在相邻两段说话人相同且后一段很短（<min_dur）时合并，减少碎片。
    不跨 speaker 合并，保证说话人边界不被稀释。
    """
    if not segments:
        return segments

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


# -------------------- 首现顺序重命名 --------------------

def renumber_speakers_by_first_appearance(segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    按首次出现顺序重命名为 SPEAKER_00, SPEAKER_01, ...
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


# -------------------- 兼容函数：整句 IoU 贴 speaker（不建议） --------------------

def attach_speaker_by_overlap(units: List[Dict[str, Any]],
                              turns: List[Tuple[float, float, str]]) -> List[Dict[str, Any]]:
    """
    旧逻辑：对整句按 IoU 贴 speaker（会把跨 speaker 的句子误贴到同一人）。
    仅作兼容保留，不在新版 finalize 流里使用。
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
            iou = inter / union if union > 0 else 0.0
            if inter > 0 and iou > best_iou:
                best_iou = iou
                best_spk = spk
        spk_to_use = best_spk if best_spk is not None else last_spk
        out.append({**u, "speaker": spk_to_use})
        if spk_to_use is not None:
            last_spk = spk_to_use
    return out


# -------------------- 总入口 --------------------

def finalize_segments(words: List[Dict[str, Any]],
                      diar_turns: List[Tuple[float, float, str]],
                      lang_hint: str = "auto") -> List[Dict[str, Any]]:
    """
    1) 合并英文碎片（修复 I sa ac -> Isaac）
    2) 逐词贴 speaker（最大重叠优先，边界更稳）
    3) 以“换人=硬切”合并成句段（结合停顿/时长）
    4) 同 speaker 短段再合并（避免过碎，不跨 speaker）
    5) 按首现顺序重命名为 SPEAKER_00/01/...
    """
    # (1) 先把英文碎片合一，避免把一个名字拆成多词导致跨段
    words = merge_ascii_fragments(words, max_gap=0.20, max_pieces=3)

    # (2) 逐词贴 speaker（最大重叠）
    words = assign_speaker_to_words(words, diar_turns)

    # (3) 合并为句段（换人=硬切）
    sents = merge_words_to_segments(words, max_gap=0.60, max_dur=8.0)

    # (4) 同 speaker 短段再合（可调整/关闭）
    sents = merge_short_segments_same_speaker(sents, min_dur=0.30)

    # (5) 首现顺序重命名（SPEAKER_00 开始）
    sents = renumber_speakers_by_first_appearance(sents)

    return sents