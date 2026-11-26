"""Unicode normalization, Indic digit folding, Devanagari->Latin phonetic transliteration, and fuzzy matching."""
from __future__ import annotations

import re
import unicodedata

# Devanagari digits U+0966..U+096F -> ASCII 0..9
_DEVA_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")

_ZERO_WIDTH_RE = re.compile(r"[\u200b-\u200f\u202a-\u202e\ufeff\u00ad]")

# Common Cyrillic/Greek homoglyphs that could disguise Latin letters
_HOMOGLYPHS = str.maketrans({
    "А": "A", "а": "a", "В": "B", "Е": "E", "е": "e", "К": "K", "к": "k",
    "М": "M", "Н": "H", "О": "O", "о": "o", "Р": "P", "р": "p", "С": "C",
    "с": "c", "Т": "T", "Х": "X", "х": "x", "І": "I", "і": "i", "Ј": "J",
    "Α": "A", "α": "a", "Β": "B", "Ε": "E", "Ζ": "Z", "Η": "H", "Ι": "I",
    "Κ": "K", "Μ": "M", "Ν": "N", "Ο": "O", "ο": "o", "Ρ": "P", "Τ": "T",
    "Χ": "X",
})

_VOWELS_INDEP = {
    "अ": "a", "आ": "aa", "इ": "i", "ई": "ee", "उ": "u", "ऊ": "oo",
    "ऋ": "ri", "ए": "e", "ऐ": "ai", "ओ": "o", "औ": "au", "ऑ": "o", "ऍ": "e",
}

_MATRAS = {
    "ा": "a", "ि": "i", "ी": "ee", "ु": "u", "ू": "oo", "ृ": "ri",
    "े": "e", "ै": "ai", "ो": "o", "ौ": "au", "ॉ": "o", "ॅ": "e",
}

_CONSONANTS = {
    "क": "k", "ख": "kh", "ग": "g", "घ": "gh", "ङ": "ng",
    "च": "ch", "छ": "chh", "ज": "j", "झ": "jh", "ञ": "ny",
    "ट": "t", "ठ": "th", "ड": "d", "ढ": "dh", "ण": "n",
    "त": "t", "थ": "th", "द": "d", "ध": "dh", "न": "n",
    "प": "p", "फ": "ph", "ब": "b", "भ": "bh", "म": "m",
    "य": "y", "र": "r", "ल": "l", "व": "v",
    "श": "sh", "ष": "sh", "स": "s", "ह": "h",
    "क़": "q", "ख़": "kh", "ग़": "gh", "ज़": "z", "ड़": "r", "ढ़": "rh", "फ़": "f",
    "ळ": "l", "ऩ": "n", "ऱ": "r", "य़": "y",
}

_NUKTA = "\u093c"
_VIRAMA = "\u094d"
_ANUSVARA = "\u0902"
_CHANDRABINDU = "\u0901"
_VISARGA = "\u0903"

_NAME_ABBREV = {
    "mohd": "mohammad",
    "md": "mohammad",
    "mohammed": "mohammad",
    "muhammad": "mohammad",
    "mohamad": "mohammad",
    "muhammed": "mohammad",
    "smt": "",
    "shri": "",
    "sri": "",
    "km": "",
    "kumari": "",
    "mr": "",
    "mrs": "",
    "ms": "",
    "dr": "",
    "ji": "",
    "sahab": "",
    "saheb": "",
    "sahib": "",
    "didi": "",
    "behenji": "",
    "bahanji": "",
    "bhai": "",
    "bhaiya": "",
    "bhaiyya": "",
    "amma": "",
    "chacha": "",
    "kaka": "",
    "babu": "",
}


def has_devanagari(text: str) -> bool:
    return any("\u0900" <= ch <= "\u097f" for ch in text)


def fold_indic_digits(text: str) -> str:
    return text.translate(_DEVA_DIGITS)


def normalize_text(text: str) -> str:
    """NFKC + zero-width strip + homoglyph fold + Indic digit fold."""
    s = unicodedata.normalize("NFKC", text)
    s = _ZERO_WIDTH_RE.sub("", s)
    s = s.translate(_HOMOGLYPHS)
    s = s.translate(_DEVA_DIGITS)
    return s


def devanagari_to_latin(text: str) -> str:
    """Rule-based Devanagari -> Hinglish/Latin transliteration for name matching."""
    text = normalize_text(text)
    out: list[str] = []
    n = len(text)
    i = 0
    while i < n:
        ch = text[i]
        # Check consonant + nukta pair
        if i + 1 < n and text[i + 1] == _NUKTA and (ch + _NUKTA) in _CONSONANTS:
            cons = _CONSONANTS[ch + _NUKTA]
            i += 2
            is_cons = True
        elif ch in _CONSONANTS:
            cons = _CONSONANTS[ch]
            i += 1
            is_cons = True
        else:
            is_cons = False

        if is_cons:
            if i < n and text[i] == _VIRAMA:
                out.append(cons)
                i += 1
            elif i < n and text[i] in _MATRAS:
                out.append(cons + _MATRAS[text[i]])
                i += 1
            else:
                # Inherent schwa 'a' unless end of word followed by nothing or punctuation
                at_word_end = (i == n) or (not ("\u0900" <= text[i] <= "\u097f"))
                if at_word_end and len(out) > 0:
                    out.append(cons)
                else:
                    out.append(cons + "a")
            continue

        if ch in _VOWELS_INDEP:
            out.append(_VOWELS_INDEP[ch])
        elif ch in (_ANUSVARA, _CHANDRABINDU):
            out.append("n")
        elif ch == _VISARGA:
            out.append("h")
        elif ch in _MATRAS:
            out.append(_MATRAS[ch])
        else:
            out.append(ch)
        i += 1
    return "".join(out)


def phonetic_key(token: str) -> str:
    """Canonical Hinglish phonetic key collapsing Indian spelling variations."""
    t = normalize_text(token)
    if has_devanagari(t):
        t = devanagari_to_latin(t)
    t = t.lower().strip(" .,'\"-_()[]{}")
    if not t:
        return ""
    if t in _NAME_ABBREV:
        t = _NAME_ABBREV[t]
        if not t:
            return ""
    # Strip Latin diacritics
    t = "".join(c for c in unicodedata.normalize("NFD", t) if unicodedata.category(c) != "Mn")
    t = re.sub(r"[^a-z0-9]", "", t)
    if not t:
        return ""
    # Common Hinglish spelling normalizations
    t = t.replace("ph", "f").replace("w", "v").replace("q", "k")
    t = t.replace("sh", "s").replace("z", "j").replace("x", "ks")
    t = t.replace("ck", "k").replace("chh", "ch")
    t = re.sub(r"([a-z])\1+", r"\1", t)  # collapse doubled consonants/vowels (hareesh -> hares, reena -> rina)
    # Collapse long vowel variants
    t = t.replace("ee", "i").replace("ea", "i").replace("oo", "u").replace("ou", "u")
    t = t.replace("ai", "e").replace("ei", "i").replace("au", "o")
    # Trailing schwa/ya variations (Pandeya vs Pandey, Rama vs Ram, Fatma vs Fatima)
    if len(t) > 4 and t.endswith("eya"):
        t = t[:-1]
    if len(t) > 4 and t.endswith("a") and not t.endswith(("ia", "ua")):
        t = t[:-1]
    # Remove non-initial short 'a'/'i' differences inside long stems lightly via consonant skeleton helper too
    return t


def consonant_skeleton(token: str) -> str:
    k = phonetic_key(token)
    if not k:
        return ""
    # Keep first char even if vowel, strip interior vowels
    return k[0] + re.sub(r"[aeiou]", "", k[1:])


def tokenize_name(name: str) -> list[str]:
    """Return normalized non-empty phonetic tokens of a person name."""
    s = normalize_text(name)
    if has_devanagari(s):
        s = devanagari_to_latin(s)
    parts = re.split(r"[\s,./]+", s.lower())
    out: list[str] = []
    for p in parts:
        pk = phonetic_key(p)
        if pk:
            out.append(pk)
    return out


def jaro_winkler(s1: str, s2: str) -> float:
    """Fast pure-Python Jaro-Winkler similarity in [0.0, 1.0]."""
    if s1 == s2:
        return 1.0
    l1, l2 = len(s1), len(s2)
    if not l1 or not l2:
        return 0.0
    max_dist = max(l1, l2) // 2 - 1
    if max_dist < 0:
        max_dist = 0
    m1 = [False] * l1
    m2 = [False] * l2
    matches = 0
    for i in range(l1):
        lo = max(0, i - max_dist)
        hi = min(i + max_dist + 1, l2)
        for j in range(lo, hi):
            if not m2[j] and s1[i] == s2[j]:
                m1[i] = True
                m2[j] = True
                matches += 1
                break
    if matches == 0:
        return 0.0
    t = 0
    k = 0
    for i in range(l1):
        if m1[i]:
            while not m2[k]:
                k += 1
            if s1[i] != s2[k]:
                t += 1
            k += 1
    jaro = (matches / l1 + matches / l2 + (matches - t / 2) / matches) / 3.0
    prefix = 0
    for c1, c2 in zip(s1[:4], s2[:4]):
        if c1 == c2:
            prefix += 1
        else:
            break
    return jaro + 0.1 * prefix * (1.0 - jaro)


def names_match(tokens_a: list[str], tokens_b: list[str]) -> float:
    """Return name match confidence in [0.0, 1.0] across Latin/Devanagari/Hinglish."""
    if not tokens_a or not tokens_b:
        return 0.0
    if tokens_a == tokens_b:
        return 1.0
    # Handle reversed order (Last First vs First Last)
    if sorted(tokens_a) == sorted(tokens_b):
        return 0.98
    # Compare single-token vs multi-token (e.g. "Anita" vs "Anita Sharma")
    if len(tokens_a) == 1 or len(tokens_b) == 1:
        t_short = tokens_a[0] if len(tokens_a) == 1 else tokens_b[0]
        other = tokens_b if len(tokens_a) == 1 else tokens_a
        if t_short == other[0] or t_short == other[-1]:
            return 0.80
        sk_short = consonant_skeleton(t_short)
        if sk_short and any(consonant_skeleton(x) == sk_short for x in other):
            return 0.75
        best = max(jaro_winkler(t_short, x) for x in other)
        return best * 0.78 if best >= 0.88 else 0.0

    # Multi-token vs multi-token: require strong match on both first and last (or skeleton)
    def tok_sim(a: str, b: str) -> float:
        if a == b:
            return 1.0
        sa, sb = consonant_skeleton(a), consonant_skeleton(b)
        if sa and sa == sb:
            return 0.94
        jw = jaro_winkler(a, b)
        return jw if jw >= 0.80 else 0.0

    # Direct alignment
    first_sim = tok_sim(tokens_a[0], tokens_b[0])
    last_sim = tok_sim(tokens_a[-1], tokens_b[-1])
    direct = min(first_sim, last_sim)
    # Swapped alignment
    swap_sim = min(tok_sim(tokens_a[0], tokens_b[-1]), tok_sim(tokens_a[-1], tokens_b[0]))
    return max(direct, swap_sim * 0.95)
