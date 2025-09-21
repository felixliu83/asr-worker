# modules/align.py
# -*- coding: utf-8 -*-

from __future__ import annotations
from typing import List, Dict, Any, Tuple, Optional
import re
import string

__all__ = [
    "merge_ascii_fragments",
    "assign_speaker_to_words",
    "fix_boundary_tokens",
    "merge_words_to_segments",
    "merge_short_segments_same_speaker",
    "revote_segment_speaker_by_overlap",
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


# -------------------- 词级边界纠偏（吸附到边界后说话人） --------------------

def fix_boundary_tokens(words: List[Dict[str, Any]],
                        turns: List[Tuple[float, float, str]],
                        max_token_dur: float = 0.22,
                        snap_window: float = 0.25) -> List[Dict[str, Any]]:
    """
    专修“句末最后一个字（极短 token）落到上一段”的问题：
      - 仅处理持续时长 <= max_token_dur 的 token
      - 若 token 的结束时间距离某个 turn 边界 <= snap_window，
        且当前 token 标注的 speaker != 边界“之后”的 speaker，则改为之后的 speaker。
    """
    if not words or not turns:
        return words

    turns_sorted = sorted(turns, key=lambda x: x[0])  # 按 start 排序
    # 构建边界：每个 turn 的结束点，以及它之后的说话人
    boundaries: List[Tuple[float, Optional[str]]] = []
    for i, (ts, te, s) in enumerate(turns_sorted):
        spk_after = turns_sorted[i + 1][2] if i + 1 < len(turns_sorted) else s
        boundaries.append((float(te), spk_after))

    def nearest_boundary(t: float):
        # 返回离时间 t 最近的边界 (tb, spk_after, dist, side)
        best = (1e9, None, 0)
        for tb, spk_after in boundaries:
            d = abs(t - tb)
            if d < best[0]:
                side = -1 if t < tb else +1  # t 在边界左/右
                best = (d, spk_after, side)
        return best

    for w in words:
        ws, we = float(w["start"]), float(w["end"])
        dur = we - ws
        if dur > max_token_dur:
            continue
        dist, spk_after, side = nearest_boundary(we)
        if dist <= snap_window and spk_after:
            if side > 0 and w.get("speaker") != spk_after:
                w["speaker"] = spk_after
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


# -------------------- 段级重投票（IoU 兜底修正） --------------------

def revote_segment_speaker_by_overlap(segments: List[Dict[str, Any]],
                                      turns: List[Tuple[float, float, str]],
                                      iou_margin: float = 0.15) -> List[Dict[str, Any]]:
    """
    对每个段，计算与各 turn 的重叠时长，取重叠总和最大的 speaker。
    若该 speaker 与当前不同，且相对优势 > iou_margin，则改写。
    用于修复整段被误标到另一个 speaker 的情况。
    """
    if not segments or not turns:
        return segments

    for seg in segments:
        us, ue = float(seg["start"]), float(seg["end"])
        overlaps: Dict[str, float] = {}
        for (ts, te, spk) in turns:
            inter = max(0.0, min(ue, te) - max(us, ts))
            if inter > 0:
                overlaps[spk] = overlaps.get(spk, 0.0) + inter

        if not overlaps:
            continue

        best_spk, best_ov = max(overlaps.items(), key=lambda x: x[1])
        cur_spk = seg.get("speaker")
        cur_ov = overlaps.get(cur_spk, 0.0)

        if cur_spk is None or best_spk != cur_spk:
            # 相对提升超过阈值才改，避免边界抖动
            if cur_ov == 0.0 or (best_ov > cur_ov * (1.0 + iou_margin)):
                seg["speaker"] = best_spk

    return segments


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
    2) 逐词贴 speaker（最大重叠优先）
    2.5) 词级边界纠偏（修“句末最后一个字跑错段”）
    3) 以“换人=硬切”合并成句段（结合停顿/时长）
    4) 同 speaker 短段再合（不跨 speaker）
    4.5) 段级 IoU 兜底重投票（修整段 speaker 误标）
    5) 按首现顺序重命名为 SPEAKER_00/01/...
    """
    # 1) 英文碎片合并
    words = merge_ascii_fragments(words, max_gap=0.20, max_pieces=3)

    # 2) 逐词贴 speaker
    words = assign_speaker_to_words(words, diar_turns)

    # 2.5) 边界纠偏（针对极短 token）
    words = fix_boundary_tokens(words, diar_turns, max_token_dur=0.22, snap_window=0.25)

    # 3) 合并为句段（换人=硬切）
    sents = merge_words_to_segments(words, max_gap=0.60, max_dur=8.0)

    # 4) 同 speaker 短段再合（避免过碎）
    sents = merge_short_segments_same_speaker(sents, min_dur=0.30)

    # 4.5) 段级 IoU 兜底“重投票”
    sents = revote_segment_speaker_by_overlap(sents, diar_turns, iou_margin=0.15)

    # 5) 首现顺序重命名
    sents = renumber_speakers_by_first_appearance(sents)

    return sents