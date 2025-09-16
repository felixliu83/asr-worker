# modules/text_clean.py
import re
from typing import Literal

# 常见口头禅
FILLERS_ZH = {"嗯","呃","额","啊","这个","那个","就是","然后","对对对","你知道","我觉得"}
FILLERS_EN = {"uh","um","erm","like","you know","i mean","sort of","kind of"}

# 中文标点映射
PUNCT_MAP_ZH = {".":"。", ",":"，", "?":"？", "!":"！", "…":"……", "....":"……", "...":"……"}

RE_MULTI_PUNCT = re.compile(r"([。，、；：？！…])\1{1,}")
RE_MULTI_DOT   = re.compile(r"\.{2,}")
RE_SPACES      = re.compile(r"\s+")

# 中英边界加空格
RE_CN_EN_GAP   = re.compile(r"([A-Za-z0-9])([\u4e00-\u9fff])|([\u4e00-\u9fff])([A-Za-z0-9])")

# CJK 空格处理
RE_SPACE_BETWEEN_CJK        = re.compile(r"(?<=[\u4e00-\u9fff])\s+(?=[\u4e00-\u9fff])")
RE_SPACE_BEFORE_CJK_PUNCT   = re.compile(r"\s+([，。？！；：、）】》」』])")
RE_SPACE_AFTER_OPEN_PUNCT   = re.compile(r"([（【《「『])\s+")

# 英文标点空格
RE_SPACE_BEFORE_EN_PUNCT    = re.compile(r"\s+([.,!?;:])")
RE_ONE_SPACE_AFTER_EN_PUNCT = re.compile(r"([.,!?;:])(?!\s|$)")

# 英文缩写（can 't -> can't / I 'm -> I'm / we'd / they're 等）
RE_EN_CONTRACTION_1 = re.compile(r"\b([A-Za-z]+)\s+'([A-Za-z]+)\b")
RE_EN_CONTRACTION_2 = re.compile(r"\b(I|you|we|they|he|she|it|that|there|can|do|does|did|is|are|was|were|has|have|had)\s+'(ll|re|ve|d|m|s|t)\b", re.I)

def _count_char_classes(text: str):
    cjk = len(re.findall(r"[\u4e00-\u9fff]", text))
    lat = len(re.findall(r"[A-Za-z]", text))
    return cjk, lat

def guess_lang(text: str) -> Literal["zh","en"]:
    cjk, lat = _count_char_classes(text)
    # 如果中文多于英文明显 → zh；否则按英文处理
    return "zh" if cjk >= max(1, int(lat*1.2)) else "en"

def normalize_punct(text: str, lang: Literal["zh","en"]) -> str:
    t = text.replace("....","……").replace("...","……")
    if lang == "zh":
        for a,b in PUNCT_MAP_ZH.items():
            t = t.replace(a,b)
        t = RE_MULTI_PUNCT.sub(r"\1", t)
        t = RE_MULTI_DOT.sub("……", t)
    else:
        # 英文不做中式替换，仅收敛多点
        t = RE_MULTI_DOT.sub("...", t)
    return t

def trim_spaces(text: str) -> str:
    t = RE_SPACES.sub(" ", text.strip())
    # 中英文之间补空格
    def _gap(m):
        g1,g2,g3,g4 = m.groups()
        if g1 and g2: return f"{g1} {g2}"
        if g3 and g4: return f"{g3} {g4}"
        return m.group(0)
    return RE_CN_EN_GAP.sub(_gap, t)

def fix_cjk_spacing(t: str) -> str:
    t = RE_SPACE_BETWEEN_CJK.sub("", t)          # 中文-中文 之间去空格
    t = RE_SPACE_BEFORE_CJK_PUNCT.sub(r"\1", t)  # 中文标点前去空格
    t = RE_SPACE_AFTER_OPEN_PUNCT.sub(r"\1", t)  # 开括号/引号后去空格
    return t

def fix_english_spacing(t: str) -> str:
    t = RE_SPACE_BEFORE_EN_PUNCT.sub(r"\1", t)                   # 标点前不留空格
    t = RE_ONE_SPACE_AFTER_EN_PUNCT.sub(r"\1 ", t)               # 标点后至少一空
    t = RE_SPACES.sub(" ", t)                                    # 收敛多空
    # 英文缩写收口
    t = RE_EN_CONTRACTION_2.sub(lambda m: f"{m.group(1)}'{m.group(2)}", t)
    t = RE_EN_CONTRACTION_1.sub(lambda m: f"{m.group(1)}'{m.group(2)}", t)
    return t.strip()

def remove_fillers(text: str, lang: Literal["zh","en"]) -> str:
    t = text
    fillers = FILLERS_ZH if lang == "zh" else FILLERS_EN
    for f in sorted(fillers, key=len, reverse=True):
        t = re.sub(rf"(?<!\S){re.escape(f)}(?!\S)", "", t, flags=re.IGNORECASE)
    return RE_SPACES.sub(" ", t).strip()

def clean_text(text: str, lang_hint: Literal["zh","en","auto"]="auto") -> str:
    lang = guess_lang(text) if lang_hint == "auto" else lang_hint
    t = normalize_punct(text, lang)
    t = remove_fillers(t, lang)
    t = trim_spaces(t)
    if lang == "zh":
        t = fix_cjk_spacing(t)
    else:
        t = fix_english_spacing(t)
    return t