# modules/align.py
# -*- coding: utf-8 -*-

from __future__ import annotations
from typing import List, Dict, Any, Tuple, Optional
import re
import string
import math

__all__ = [
    "finalize_segments",
    "merge_ascii_fragments",
    "assign_speaker_to_words",
    "smooth_word_speakers",
    "fix_boundary_tokens",
    "merge_words_to_segments",
    "merge_short_segments_same_speaker",
    "absorb_jitter_switches",
    "revote_segment_speaker_by_overlap",
    "revote_segment_by_word_majority",
    "renumber_speakers_by_first_appearance",
    "attach_speaker_by_overlap",  # 兼容旧调用
]

# -------------------- 可调参数（按需微调） --------------------

# 英文碎片合并
ASCII_MAX_GAP = 0.20     # 两英文碎片最大间隔
ASCII_MAX_PIECES = 3     # 至多合并几段碎片

# 词层说话人贴标
BOUNDARY_SNAP_WINDOW = 0.35   # 秒，距离 turn 边界这么近就“吸附到边界后”
BOUNDARY_MAX_TOKEN_DUR = 0.28 # 秒，仅修极短 token（避免过度）
SMOOTH_WINDOW_SEC = 0.40      # 秒，词层滑窗多数票平滑窗口

# 切段（词→句）
MAX_INTER_WORD_GAP = 0.60     # 大于此停顿必切
MAX_SEG_DUR = 8.0             # 段最长时长，强制切
PUNCT_BREAK_RE = re.compile(r"[。！？!?]|(\.\s)|[.…]+")

# 片段后处理
SAME_SPK_MERGE_SHORT = 0.30   # 相邻同speaker且后一段<此阈值，则合并
JITTER_ABSORB_DUR = 0.18      # 小于此阈值的“瞬切”尝试吸回

# 片段级重投票
IOU_MARGIN = 0.08             # 新候选对当前speaker的相对优势阈值
WORD_MAJORITY_RATIO = 0.60    # 片段内词多数票占比阈值（超过才改）

# ------------------------------------------------------------

# 简单中英混排拼接：ASCII-ASCII 之间加空格
def _smart_join(prev: str, token: str) -> str:
    if not prev:
        return token
    a = prev[-1]
    b = token[:1]
    if a.isascii() and b.isascii():
        return prev + " " + token
    return prev + token

def _is_ascii_alpha(s: str) -> bool:
    return bool(s) and all(ch in string.ascii_letters for ch in s)

# -------------------- 1) 英文碎片合并 --------------------

def merge_ascii_fragments(words: List[Dict[str, Any]],
                          max_gap: float = ASCII_MAX_GAP,
                          max_pieces: int = ASCII_MAX_PIECES) -> List[Dict[str, Any]]:
    if not words:
        return words
    out: List[Dict[str, Any]] = []
    i, n = 0, len(words)
    while i < n:
        w = words[i]
        txt = w["text"]
        ws, we = float(w["start"]), float(w["end"])
        if len(txt) == 1 and txt.isupper() and _is_ascii_alpha(txt):
            pieces = [txt]; start, end = ws, we; j = i + 1
            while j < n and (len(pieces) - 1) < max_pieces:
                nxt = words[j]
                gap = float(nxt["start"]) - end
                t = nxt["text"]
                if gap <= max_gap and _is_ascii_alpha(t) and t.islower() and 1 <= len(t) <= 3:
                    pieces.append(t); end = float(nxt["end"]); j += 1
                else:
                    break
            if len(pieces) >= 2:
                merged = pieces[0] + "".join(pieces[1:])
                merged = merged[0] + merged[1:].lower()
                out.append({"start": start, "end": end, "text": merged})
                i = j; continue
        out.append(w); i += 1
    return out

# -------------------- 2) 词层说话人：最大重叠 + 平滑 + 边界吸附 --------------------

def assign_speaker_to_words(words: List[Dict[str, Any]],
                            turns: List[Tuple[float, float, str]]) -> List[Dict[str, Any]]:
    if not turns:
        for w in words: w["speaker"] = None
        return words
    for w in words:
        ws, we = float(w["start"]), float(w["end"])
        best_overlap, best_spk = 0.0, None
        for (ts, te, s) in turns:
            inter = max(0.0, min(we, te) - max(ws, ts))
            if inter > best_overlap:
                best_overlap, best_spk = inter, s
        if best_spk is None:
            # 最近端点兜底（比 midpoint 稳）
            mid = 0.5 * (ws + we); best = (1e9, None)
            for (ts, te, s) in turns:
                d = 0.0 if ts <= mid <= te else min(abs(mid - ts), abs(mid - te))
                if d < best[0]: best = (d, s)
            best_spk = best[1]
        w["speaker"] = best_spk
    return words

def smooth_word_speakers(words: List[Dict[str, Any]],
                         window_sec: float = SMOOTH_WINDOW_SEC) -> List[Dict[str, Any]]:
    """滑窗多数票平滑词层speaker，抑制短周期抖动"""
    if not words or window_sec <= 0:
        return words
    times = [0.5*(float(w["start"])+float(w["end"])) for w in words]
    L = len(words)
    j_left = 0
    for i, mid in enumerate(times):
        # 滑窗 [mid - w/2, mid + w/2]
        left, right = mid - window_sec/2, mid + window_sec/2
        while j_left < L and times[j_left] < left:
            j_left += 1
        j_right = i
        while j_right+1 < L and times[j_right+1] <= right:
            j_right += 1
        counts = {}
        for k in range(j_left, j_right+1):
            spk = words[k].get("speaker")
            if spk is not None:
                counts[spk] = counts.get(spk, 0) + 1
        if counts:
            best_spk = max(counts.items(), key=lambda x: x[1])[0]
            words[i]["speaker"] = best_spk
    return words

def fix_boundary_tokens(words: List[Dict[str, Any]],
                        turns: List[Tuple[float, float, str]],
                        max_token_dur: float = BOUNDARY_MAX_TOKEN_DUR,
                        snap_window: float = BOUNDARY_SNAP_WINDOW) -> List[Dict[str, Any]]:
    if not words or not turns: return words
    turns_sorted = sorted(turns, key=lambda x: x[0])
    boundaries: List[Tuple[float, Optional[str]]] = []
    for i, (ts, te, s) in enumerate(turns_sorted):
        spk_after = turns_sorted[i+1][2] if i+1 < len(turns_sorted) else s
        boundaries.append((float(te), spk_after))
    def nearest_boundary(t: float):
        best = (1e9, None, 0)
        for tb, spk_after in boundaries:
            d = abs(t - tb)
            if d < best[0]:
                side = -1 if t < tb else +1
                best = (d, spk_after, side)
        return best
    for w in words:
        ws, we = float(w["start"]), float(w["end"]); dur = we - ws
        if dur > max_token_dur: continue
        dist, spk_after, side = nearest_boundary(we)
        if dist <= snap_window and spk_after and side > 0 and w.get("speaker") != spk_after:
            w["speaker"] = spk_after
    return words

# -------------------- 3) 词→句：换人=硬切 --------------------

def merge_words_to_segments(words: List[Dict[str, Any]],
                            *,
                            max_gap: float = MAX_INTER_WORD_GAP,
                            max_dur: float = MAX_SEG_DUR) -> List[Dict[str, Any]]:
    if not words: return []
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
        spk = w.get("speaker"); txt = w["text"]
        if cur is None:
            cur = {"start": ws, "end": we, "speaker": spk, "text": txt}
            last_end = we; continue
        # 硬切：换人
        if spk != cur.get("speaker"):
            flush(); cur = {"start": ws, "end": we, "speaker": spk, "text": txt}
            last_end = we; continue
        # 切：停顿过大
        gap = ws - (last_end if last_end is not None else ws)
        if gap > max_gap:
            flush(); cur = {"start": ws, "end": we, "speaker": spk, "text": txt}
            last_end = we; continue
        # 追加
        cur["end"] = we
        cur["text"] = _smart_join(cur["text"], txt)
        # 软切：句末标点 & 段>=1.5s
        if PUNCT_BREAK_RE.search(txt) and (cur["end"] - cur["start"]) >= 1.5:
            flush()
        # 强切：过长
        if cur and (cur["end"] - cur["start"]) >= max_dur:
            flush()
        last_end = we
    flush(); return segs

# -------------------- 4) 片段后处理：合并与抖动吸收 --------------------

def merge_short_segments_same_speaker(segments: List[Dict[str, Any]],
                                      min_dur: float = SAME_SPK_MERGE_SHORT) -> List[Dict[str, Any]]:
    if not segments: return segments
    out: List[Dict[str, Any]] = []
    for seg in segments:
        if not out:
            out.append(seg); continue
        dur = float(seg["end"]) - float(seg["start"])
        if dur < min_dur and seg.get("speaker") == out[-1].get("speaker"):
            out[-1]["end"] = seg["end"]
            out[-1]["text"] = _smart_join(out[-1]["text"], seg["text"])
        else:
            out.append(seg)
    return out

def absorb_jitter_switches(segments: List[Dict[str, Any]],
                           max_jitter: float = JITTER_ABSORB_DUR) -> List[Dict[str, Any]]:
    """
    处理 A|B|A 里非常短的 B：若 B < max_jitter，则吸收到 A（避免抖动）
    """
    if len(segments) < 3: return segments
    out = segments[:]
    i = 1
    while i+1 < len(out):
        prev, cur, nxt = out[i-1], out[i], out[i+1]
        cur_dur = float(cur["end"]) - float(cur["start"])
        if (cur.get("speaker") != prev.get("speaker")
            and nxt.get("speaker") == prev.get("speaker")
            and cur_dur < max_jitter):
            # 吸收：并到前段（也可选择并到后段）
            prev["end"] = nxt["start"]  # 截断到抖动开始
            # 删除抖动段
            out.pop(i)     # remove cur
            # 合并 prev 与 nxt
            prev["end"] = nxt["end"]
            prev["text"] = _smart_join(prev["text"], nxt["text"])
            out.pop(i)     # remove nxt
            # 不递增 i，继续检查新的三元组
            continue
        i += 1
    return out

# -------------------- 5) 片段重投票（IoU + 词多数） --------------------

def revote_segment_speaker_by_overlap(segments: List[Dict[str, Any]],
                                      turns: List[Tuple[float, float, str]],
                                      iou_margin: float = IOU_MARGIN) -> List[Dict[str, Any]]:
    if not segments or not turns: return segments
    for seg in segments:
        us, ue = float(seg["start"]), float(seg["end"])
        overlaps: Dict[str, float] = {}
        for (ts, te, spk) in turns:
            inter = max(0.0, min(ue, te) - max(us, ts))
            if inter > 0: overlaps[spk] = overlaps.get(spk, 0.0) + inter
        if not overlaps: continue
        best_spk, best_ov = max(overlaps.items(), key=lambda x: x[1])
        cur_spk = seg.get("speaker"); cur_ov = overlaps.get(cur_spk, 0.0)
        if cur_spk is None or best_spk != cur_spk:
            if cur_ov == 0.0 or (best_ov > cur_ov * (1.0 + iou_margin)):
                seg["speaker"] = best_spk
    return segments

def revote_segment_by_word_majority(segments: List[Dict[str, Any]],
                                    words: List[Dict[str, Any]],
                                    majority_ratio: float = WORD_MAJORITY_RATIO) -> List[Dict[str, Any]]:
    if not segments or not words: return segments
    widx, n = 0, len(words)
    out = []
    for seg in segments:
        us, ue = float(seg["start"]), float(seg["end"])
        # 推进到可能相交的首词
        while widx < n and float(words[widx]["end"]) <= us:
            widx += 1
        j = widx; counts = {}
        while j < n and float(words[j]["start"]) < ue:
            spk = words[j].get("speaker")
            if spk is not None: counts[spk] = counts.get(spk, 0) + 1
            j += 1
        if counts:
            best_spk, best_cnt = max(counts.items(), key=lambda x: x[1])
            total = sum(counts.values())
            cur_spk = seg.get("speaker")
            if best_spk != cur_spk and best_cnt >= majority_ratio * total:
                seg = {**seg, "speaker": best_spk}
        out.append(seg)
    return out

# -------------------- 6) 首现顺序重命名 --------------------

def renumber_speakers_by_first_appearance(segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    mapping: Dict[str, str] = {}; nxt = 0
    for seg in segments:
        spk = seg.get("speaker")
        if spk is None: continue
        if spk not in mapping:
            mapping[spk] = f"SPEAKER_{nxt:02d}"; nxt += 1
        seg["speaker"] = mapping[spk]
    return segments

# -------------------- 兼容：整句 IoU 贴 speaker --------------------

def attach_speaker_by_overlap(units: List[Dict[str, Any]],
                              turns: List[Tuple[float, float, str]]) -> List[Dict[str, Any]]:
    if not turns:
        return [{**u, "speaker": None} for u in units]
    out = []; last_spk: Optional[str] = None
    for u in units:
        us, ue = float(u["start"]), float(u["end"])
        best_iou, best_spk = 0.0, None
        for (ts, te, spk) in turns:
            inter = max(0.0, min(ue, te) - max(us, ts))
            uni = max(ue, te) - min(us, ts)
            iou = (inter/uni) if uni > 0 else 0.0
            if inter > 0 and iou > best_iou:
                best_iou, best_spk = iou, spk
        spk_use = best_spk if best_spk is not None else last_spk
        out.append({**u, "speaker": spk_use})
        if spk_use is not None: last_spk = spk_use
    return out

# -------------------- 入口：finalize --------------------

def finalize_segments(words: List[Dict[str, Any]],
                      diar_turns: List[Tuple[float, float, str]],
                      lang_hint: str = "auto") -> List[Dict[str, Any]]:
    # 0) 英文碎片合并（避免名字/拼写被拆）
    words = merge_ascii_fragments(words, max_gap=ASCII_MAX_GAP, max_pieces=ASCII_MAX_PIECES)
    # 1) 词层贴 speaker（最大重叠）→ 滑窗平滑 → 边界吸附
    words = assign_speaker_to_words(words, diar_turns)
    words = smooth_word_speakers(words, window_sec=SMOOTH_WINDOW_SEC)
    words = fix_boundary_tokens(words, diar_turns,
                                max_token_dur=BOUNDARY_MAX_TOKEN_DUR,
                                snap_window=BOUNDARY_SNAP_WINDOW)
    # 2) 词→句（换人=硬切 + 停顿/时长）
    sents = merge_words_to_segments(words,
                                    max_gap=MAX_INTER_WORD_GAP,
                                    max_dur=MAX_SEG_DUR)
    # 3) 片段后处理：同speaker短段合并 + 抖动吸收
    sents = merge_short_segments_same_speaker(sents, min_dur=SAME_SPK_MERGE_SHORT)
    sents = absorb_jitter_switches(sents, max_jitter=JITTER_ABSORB_DUR)
    # 4) 片段重投票：turn IoU + 词多数双兜底
    sents = revote_segment_speaker_by_overlap(sents, diar_turns, iou_margin=IOU_MARGIN)
    sents = revote_segment_by_word_majority(sents, words, majority_ratio=WORD_MAJORITY_RATIO)
    # 5) 重命名
    sents = renumber_speakers_by_first_appearance(sents)
    return sents