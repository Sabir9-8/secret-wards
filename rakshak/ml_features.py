"""Shared deterministic feature extractor for span-level PII vs. clinical disambiguation.

Uses ONLY Python standard library (`re`, `unicodedata`, `math`).
"""
from __future__ import annotations

import math
import re
import unicodedata

_DEVA_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")
_ZW_RE = re.compile(r"[\u200b-\u200f\u202a-\u202e\ufeff\u00ad]")
_HOMOGLYPHS = str.maketrans({
    "А": "A", "а": "a", "В": "B", "Е": "E", "е": "e", "К": "K", "к": "k",
    "М": "M", "Н": "H", "О": "O", "о": "o", "Р": "P", "р": "p", "С": "C",
    "с": "c", "Т": "T", "Х": "X", "х": "x", "І": "I", "і": "i", "Ј": "J",
    "Α": "A", "α": "a", "Β": "B", "Ε": "E", "Ζ": "Z", "Η": "H", "Ι": "I",
    "Κ": "K", "Μ": "M", "Ν": "N", "Ο": "O", "ο": "o", "Ρ": "P", "Τ": "T",
    "Χ": "X",
})

_WS_RE = re.compile(r"\s+")
# Keep Devanagari combining matras/virama/anusvara inside word tokens
_TOKEN_RE = re.compile(r"(?:[^\W_]|[\u0900-\u097f])+", re.UNICODE)

# Context regexes
_LEFT_DR_RE = re.compile(r"(?:^|\b)(?:dr\.?|डॉ\.?|doctor|consultant|physician|surgeon)\s*$", re.I)
_LEFT_KCO_HO_RE = re.compile(r"(?:^|\b)(?:k/c/o|h/o)\s*:?\s*$", re.I)
_LEFT_KCO_HO_NEAR_RE = re.compile(r"(?:^|\b)(?:k/c/o|h/o)\s*:?\s*[^.;\n]{0,30}$", re.I)
_LEFT_CO_RAW_RE = re.compile(r"(?:^|(?<![kh]/))\bc/o\s*:?\s*$", re.I)
_LEFT_REL_RE = re.compile(
    r"(?:^|\b)(?:s/[o0]|d/[o0]|w/[o0]|attendant|guardian|caregiver|father|husband|mother|spouse|next\s+of\s+kin|पुत्र|पुत्री|पत्नी|पिता(?:\s+का\s+नाम)?|पति(?:\s+का\s+नाम)?|माता)\s*[:.]?\s*$",
    re.I,
)
_LEFT_PII_LABEL_RE = re.compile(
    r"(?:^|\b)(?:patient(?:'s)?(?:\s+name)?|pt\.?(?:\s+name)?|beneficiary(?:\s+name)?|name|naam|address|pata|vill(?:age)?|gaon|gram|mohalla|locality|landmark|near|opp\.?|behind|mob(?:ile)?|ph(?:one)?|contact|pin(?:\s*code)?|zip|uhid|mrn|नाम|मरीज़\s+का\s+नाम|पता|ग्राम|गाँव|मोहल्ला|पिन|मोबाइल)\s*[:：-]?\s*$",
    re.I,
)
_LEFT_CLINICAL_RE = re.compile(
    r"(?:^|\b)(?:tab\.?|cap\.?|inj\.?|syp\.?|inh\.?|rx|advised|continue|diagnosis|differential|complaints|history|medication(?:s)?|vitals|labs?|monitor|plan|treatment|निदान|दवा)\s*[:.]?\s*$",
    re.I,
)
_LEFT_WARD_RE = re.compile(r"(?:^|\b)(?:ward|bed|icu|opd|ipd|department|dept\.?)\s*[:#-]?\s*$", re.I)

_RIGHT_HONORIFIC_RE = re.compile(
    r"^\s*(?:ji|sahab|saheb|sahib|didi|behenji|bahanji|bhaiya|bhai|amma|chacha|kaka|babu|जी|साहब|दीदी|बहनजी|भैया|अम्मा|ka|ki|ke|ko|ne|का|की|के|को|ने)\b",
    re.I,
)
_RIGHT_EPONYM_RE = re.compile(
    r"^\s*(?:'s\s+|s'\s+)?(?:disease|syndrome|palsy|lymphoma|triad|sign|phenomenon|crisis|catheter|lactate|abdomen|test|fracture|reflex|score|criteria|solution|cannula|tube|maneuver|scale)\b",
    re.I,
)
_RIGHT_CLINICAL_UNIT_RE = re.compile(
    r"^\s*(?:mg/dl|g/dl|mmol/l|miu/l|/ul|mmhg|/min|mg|mcg|ml|iu|g|%|bd|od|tds|qid|sos|iv|im|po|days?|wks?|weeks?|months?|hrs?|x\s*\d+|[01]-[01]-[01])\b",
    re.I,
)
_RIGHT_DOCTOR_DEG_RE = re.compile(
    r"^\s*(?:\((?:md|ms|mbbs|dm|dnb|mch|frcs)[^)]*\)|reviewed|advised|examined|prescribed|on\s+round|took\s+round|is\s+the\s+nodal)\b",
    re.I,
)
_RIGHT_PERSON_CTX_RE = re.compile(
    r"^\s*(?:\((?:attendant|visitor|patient|relative|guardian|caregiver)|,\s*(?:s/o|d/o|w/o|c/o|\d{1,3}\s*(?:y|yrs?|वर्ष)?\s*/\s*[mf])|\bs/o\b|\bd/o\b|\bw/o\b)",
    re.I,
)
_RIGHT_PIN_RE = re.compile(r"^\s*[,-]?\s*[1-8]\d{5}\b")
_LEFT_ADDR_RE = re.compile(r"(?:^|\b)(?:address|pata|location|city|town|vill(?:age)?|gaon|gram|mohalla|locality|पता|ग्राम|गाँव|मोहल्ला)\s*[:：-]?\s*$", re.I)

# Span self-cues
_EPONYM_WORDS = {
    "disease", "syndrome", "palsy", "lymphoma", "triad", "sign", "phenomenon",
    "crisis", "catheter", "lactate", "abdomen", "fracture", "reflex", "criteria",
    "ward", "bed", "hospital", "clinic", "medical", "protocol", "guidelines",
    "tablet", "capsule", "injection", "syrup", "inhaler", "fever", "cough",
    "pain", "chills", "diabetes", "asthma", "hypertension", "hepatitis",
    "typhoid", "dengue", "malaria", "anaemia", "anemia", "tuberculosis",
    "hypothyroidism", "gastritis", "pneumonia", "infection", "t2dm", "ptb",
    "creatinine", "haemoglobin", "hemoglobin", "hba1c", "wbc", "spo2", "bp",
    "dolo", "paracetamol", "metformin", "ramipril", "amlodipine", "pantoprazole",
    "ceftriaxone", "doxycycline", "salbutamol", "levothyroxine", "azithromycin",
    "बुखार", "टीबी", "निमोनिया", "मलेरिया", "संक्रमण", "दवा", "वार्ड", "अस्पताल",
}
_LOC_WORDS = {
    "mohalla", "nagar", "colony", "bazar", "bazaar", "ganj", "pur", "purwa",
    "kalan", "khurd", "gaon", "gram", "tehsil", "mandi", "road", "lane",
    "street", "marg", "chowk", "vihar", "enclave", "sector", "block", "bagh",
    "near", "opp", "opposite", "behind", "mandir", "masjid", "gurudwara",
    "panchayat", "bhawan", "tanki", "stand", "station",
    "मोहल्ला", "नगर", "ग्राम", "गाँव", "गली", "कॉलोनी", "रोड", "कलां", "खुर्द", "वार्ड", "तहसील", "जिला",
}
_HONORIFIC_WORDS = {
    "ji", "sahab", "saheb", "sahib", "didi", "behenji", "bahanji", "bhaiya",
    "bhai", "amma", "chacha", "kaka", "babu", "shri", "smt", "kumari",
    "जी", "साहब", "दीदी", "बहनजी", "भैया", "अम्मा", "श्री", "श्रीमती",
}
_DOSE_VITAL_RE = re.compile(
    r"(?i)(?:^[01]-[01]-[01]$|^\d{2,3}/\d{2,3}$|^(?:bp|spo2|hb|hba1c|wbc|icd-?10)\b|\b\d+(?:\.\d+)?\s*(?:mg/dl|g/dl|mmol/l|miu/l|/ul|mmhg|/min|mg|mcg|ml|%|°?[fc])\b|\b[a-tv-z]\d{2}(?:\.\d{1,2})?\b|\b(?:bd|od|tds|qid|sos)\s*x\s*\d+\b)"
)
_HOUSE_RE = re.compile(r"(?i)\b(?:h\.?\s*no\.?|house|makan|flat|plot|मकान\s*नं)\b")
_TITLE_MULTI_RE = re.compile(r"^[A-Z][a-zA-Z.'-]*(?:\s+[A-Z][a-zA-Z.'-]*){1,3}$")
_TITLE_SINGLE_RE = re.compile(r"^[A-Z][a-z]{2,}$")


def normalize_for_ml(text: str) -> str:
    """NFKC, strip zero-width chars, fold homoglyphs and Indic digits, collapse whitespace."""
    s = unicodedata.normalize("NFKC", text)
    s = _ZW_RE.sub("", s)
    s = s.translate(_HOMOGLYPHS)
    s = s.translate(_DEVA_DIGITS)
    return _WS_RE.sub(" ", s).strip()


def extract_features(span: str, left_ctx: str = "", right_ctx: str = "") -> tuple[dict[str, float], dict[str, float]]:
    """Return (ngram_counts, struct_features) deterministically for a candidate span."""
    s = normalize_for_ml(span)
    lc = normalize_for_ml(left_ctx[-80:]) if left_ctx else ""
    rc = normalize_for_ml(right_ctx[:80]) if right_ctx else ""

    ngrams: dict[str, float] = {}
    tokens = _TOKEN_RE.findall(s.casefold())
    for tok in tokens:
        if not tok:
            continue
        bw = f"<{tok}>"
        L = len(bw)
        for n in (2, 3, 4):
            if L < n:
                continue
            for i in range(L - n + 1):
                g = f"ng:{bw[i : i + n]}"
                ngrams[g] = ngrams.get(g, 0.0) + 1.0

    # Add lightweight context affixes from immediate neighbor tokens (abstracting numbers)
    def _ctx_tok(t: str) -> str:
        if t.isdigit():
            if len(t) == 6 and t[0] in "12345678":
                return "#pin6"
            if len(t) >= 10:
                return "#phone"
            return "#num"
        return t

    l_toks = _TOKEN_RE.findall(lc.casefold())[-2:]
    for lt in l_toks:
        k = f"lc:{_ctx_tok(lt)}"
        ngrams[k] = ngrams.get(k, 0.0) + 1.0
    r_toks = _TOKEN_RE.findall(rc.casefold())[:2]
    for rt in r_toks:
        k = f"rc:{_ctx_tok(rt)}"
        ngrams[k] = ngrams.get(k, 0.0) + 1.0

    struct: dict[str, float] = {}

    def _set(name: str, cond: bool) -> None:
        if cond:
            struct[f"st:{name}"] = 1.0

    # Left context features
    _set("left_dr", bool(_LEFT_DR_RE.search(lc)))
    _set("left_kco_ho", bool(_LEFT_KCO_HO_RE.search(lc) or _LEFT_KCO_HO_NEAR_RE.search(lc)))
    _set("left_co_raw", bool(_LEFT_CO_RAW_RE.search(lc)))
    _set("left_rel_marker", bool(_LEFT_REL_RE.search(lc)))
    _set("left_pii_label", bool(_LEFT_PII_LABEL_RE.search(lc)))
    _set("left_addr_label", bool(_LEFT_ADDR_RE.search(lc)))
    _set("left_clinical", bool(_LEFT_CLINICAL_RE.search(lc)))
    _set("left_ward", bool(_LEFT_WARD_RE.search(lc)))

    # Right context features
    _set("right_honorific", bool(_RIGHT_HONORIFIC_RE.search(rc)))
    _set("right_eponym", bool(_RIGHT_EPONYM_RE.search(rc)))
    _set("right_clinical_unit", bool(_RIGHT_CLINICAL_UNIT_RE.search(rc)))
    _set("right_doctor_deg", bool(_RIGHT_DOCTOR_DEG_RE.search(rc)))
    _set("right_person_ctx", bool(_RIGHT_PERSON_CTX_RE.search(rc)))
    _set("right_pin", bool(_RIGHT_PIN_RE.search(rc)))

    # Span morphology & vocabulary features
    has_deva = any("\u0900" <= ch <= "\u097f" for ch in s)
    has_latin = any("a" <= ch.lower() <= "z" for ch in s)
    has_digits = any(ch.isdigit() for ch in s)
    toks_cf = set(tokens)

    _set("has_deva", has_deva)
    _set("has_latin", has_latin)
    _set("has_digits", has_digits)
    _set("has_apostrophe_s", s.endswith(("'s", "s'", "’s")) or "'s " in s.lower())
    _set("contains_clinical_word", bool(toks_cf & _EPONYM_WORDS))
    _set("contains_loc_word", bool(toks_cf & _LOC_WORDS) or bool(_HOUSE_RE.search(s)))
    _set("contains_honorific", bool(toks_cf & _HONORIFIC_WORDS))
    _set("shape_dose_vital", bool(_DOSE_VITAL_RE.search(s)))
    _set("shape_title_single", bool(_TITLE_SINGLE_RE.match(s)))
    _set("shape_title_multi", bool(_TITLE_MULTI_RE.match(s)))
    _set("shape_all_lower", bool(s) and has_latin and s == s.lower() and not has_digits)
    _set("shape_all_upper", bool(s) and has_latin and s == s.upper() and len(s) <= 8)
    _set("wc_1", len(tokens) == 1)
    _set("wc_2", len(tokens) == 2)
    _set("wc_3plus", len(tokens) >= 3)

    return ngrams, struct


def vectorize_sparse(
    ngrams: dict[str, float],
    struct: dict[str, float],
    vocab: dict[str, int],
    idf: list[float],
) -> dict[int, float]:
    """Convert raw n-gram & structural feature dicts into an indexed sparse vector.

    N-gram TF-IDF features are L2-normalized independently so span length stays balanced,
    while structural binary features keep unit scale.
    """
    vec: dict[int, float] = {}
    ng_sq = 0.0
    ng_tmp: list[tuple[int, float]] = []
    for k, tf in ngrams.items():
        idx = vocab.get(k)
        if idx is not None:
            val = tf * idf[idx]
            ng_tmp.append((idx, val))
            ng_sq += val * val
    if ng_sq > 0.0:
        inv_norm = 1.0 / math.sqrt(ng_sq)
        for idx, val in ng_tmp:
            vec[idx] = val * inv_norm

    for k, val in struct.items():
        idx = vocab.get(k)
        if idx is not None:
            vec[idx] = val * idf[idx]
    return vec
