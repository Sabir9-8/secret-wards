"""Context-aware multilingual PII detection, meaning-trap protection, date/age parsing, and channel scrubbing."""
from __future__ import annotations

import base64
import binascii
import datetime as dt
import re
import urllib.parse
from dataclasses import dataclass, field
from typing import Any

from .gazetteer import (
    DEVANAGARI_NAMES,
    DISEASE_EPONYM_SUFFIXES,
    HONORIFIC_SUFFIXES,
    HOSPITAL_MARKERS,
    INDIAN_FIRST_NAMES,
    INDIAN_SURNAMES,
    NON_PII_WORDS,
    TITLE_PREFIXES,
)
from .policy import age_band, date_shift_days, generalize_pin, hmac_token
from .translit import devanagari_to_latin, fold_indic_digits, has_devanagari, normalize_text, phonetic_key, tokenize_name

# --------------------------------------------------------------------------- Patterns
_EMAIL_RE = re.compile(r"(?<![A-Za-z0-9_.+-])[A-Za-z0-9_.+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+(?![A-Za-z0-9_.+-])")
_URL_PCT_RE = re.compile(r"%[0-9A-Fa-f]{2}")
_ABHA_ADDR_RE = re.compile(r"\b[\w.+-]+@(?:abdm|ndhm|sbx)\b", re.I)
_ABHA_NUM_RE = re.compile(r"(?<![-\d०-९])[\d०-९]{2}[-\s]?[\d०-९]{4}[-\s]?[\d०-९]{4}[-\s]?[\d०-९]{4}(?![-\d०-९])")
_AADHAAR_RE = re.compile(r"(?<![-\d०-९])[\d०-९]{4}[-\s]?[\d०-९]{4}[-\s]?[\d०-९]{4}(?![-\d०-९])")
_AADHAAR_MASKED_RE = re.compile(r"\b[xX*]{4}[-\s]?[xX*]{4}[-\s]?[\d०-९]{4}\b")
_PAN_RE = re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b")
_INSURANCE_RE = re.compile(r"\b(?:PMJAY[-/][A-Z0-9-]+|POL/\d{2}/\d{4,})\b", re.I)
_MRN_RE = re.compile(
    r"(?<![A-Za-z0-9])(?:"
    r"[A-Z]{2,6}(?:[-/][A-Z0-9]{1,5}){1,3}[-/]\d{3,}"
    r"|(?:SNJ|JJH|DHH|UHID|IPD|ADM|LGC|MRN)[-/]?\d{3,}(?:[-/]\d+)*"
    r"|DHH\d{4}-\d{4,}"
    r"|(?:UHID|ADM)\d{4,}"
    r")(?![A-Za-z0-9])",
    re.I,
)
_MRN_LABELED_RE = re.compile(
    r"(?i)(\b(?:UHID|Reg\.?\s*No\.?|IPD\s*No\.?|MRN|Patient\s*ID|पंजीकरण\s*सं\.?|आईपीडी\s*सं\.?)\s*[:#-]?\s*)([A-Z0-9][A-Z0-9/-]{3,})"
)
_PHONE_RE = re.compile(
    r"(?<![\d०-९])(?:\+?91[-\s]?|0)?[6-9६-९][\d०-९]{4}[-\s]?[\d०-९]{5}(?![\d०-९])"
)
_PHONE_MASKED_RE = re.compile(r"\b[6-9]\d[xX*]{4,6}\d{2,4}\b")
_PHONE_LABELED_RE = re.compile(
    r"(?i)((?:\b(?:mob(?:ile)?|ph(?:one)?|call(?:\s*karo|\s*back)?|callback|tel)|मोबाइल|फ़ोन)\s*[:.]?\s*)((?:\+?91[-\s]?)?[0-9०-९][0-9०-९ \t-]{5,14}[0-9०-९])"
)

# Addresses / house / ward / mohalla / landmark / village / PIN
_HOUSE_RE = re.compile(
    r"(?i)(?:\b(?:H\.?\s*No\.?|House(?:\s*No\.?)?|Makan\s*(?:No\.?)?|Flat\s*(?:No\.?)?|Plot\s*(?:No\.?)?)\s*[:.]?\s*[A-Z0-9०-९/-]+|मकान\s*नं\.?\s*[0-9०-९/-]+)"
)
_MOHALLA_RE = re.compile(
    r"(?i)\bMohalla[ \t]+[A-Za-z0-9 \t]+?(?=,|\.|;|\n|\b(?:near|opp\.?|opposite|behind|Tehsil|Distt|District)\b|$)"
)
_RES_WARD_RE = re.compile(
    r"(?i)(?:\bWard[ \t]+\d+(?=\s*,|\s+Mohalla|\s+near|\s+opp|\s+behind|\s+Vill|\s+Gram|\s+Tehsil)|(?<=,\s)Ward[ \t]+\d+(?=\s*(?:$|\n|,)))"
)
_LANDMARK_RE = re.compile(
    r"(?i)(?:\b(?:near|opp\.?|opposite|behind)[ \t]+[A-Za-z0-9 \t]+?(?=,|\.|;|\n|\b(?:Tehsil|Distt|District|Vill|Gram)\b|$)|[\u0900-\u097f \t]+?[ \t]के[ \t](?:पीछे|सामने|पास))"
)
_VILLAGE_ANCHORED_RE = re.compile(
    r"(?i)(?:(?:\bVill\.?|\bVillage|\bGaon|\bGram)\s*[:.]?\s*[A-Za-z][A-Za-z \t]{1,35}?(?=,|\.|;|\n|\b(?:Tehsil|Distt|District)\b|$)|(?:ग्राम|गाँव)\s*[:.]?\s*[\u0900-\u097f]+(?:[ \t]+[\u0900-\u097f]+){0,2}(?=,|\.|;|\n|तहसील|जिला|$))"
)
_ADDR_VILLAGE_BEFORE_TEHSIL_RE = re.compile(
    r"(?i)(,\s*)([A-Z][a-z]+(?:[ \t]+[A-Z][a-z]+){0,2})(?=\s*,\s*Tehsil\b)"
)
_TEHSIL_ANCHORED_RE = re.compile(
    r"(?i)(?:\bTehsil[ \t]+[A-Za-z]+(?:[ \t]+[A-Za-z]+)?|तहसील[ \t]+[\u0900-\u097f]+)"
)
_PIN_LABELED_RE = re.compile(
    r"(?i)((?:PIN(?:\s*Code)?|पिन(?:\s*कोड)?|Zip(?:\s*Code)?|Postal\s*Code)\s*[:：-]?\s*|[-,\\/]\s*)([1-8][\d०-९]{5})(?![\d०-९])"
)
_WIN_PATH_RE = re.compile(r"\b[A-Za-z]:\\[^\s\"',;)]+")

# Protected clinical spans (shielded before name/number/PIN scrubbing)
_DR_LATIN_RE = re.compile(r"\bDr\.?[ \t]+(?:[A-Z][a-zA-Z.'-]+(?:[ \t]+[A-Z][a-zA-Z.'-]+){0,3})")
_DR_DEVA_RE = re.compile(r"डॉ\.?[ \t]*[\u0900-\u097f]+(?:[ \t]+[\u0900-\u097f]+){0,3}")
_EPONYM_RE = re.compile(
    r"\b[A-Za-z]+(?:'s|s')?[ \t]+(?:" + "|".join(DISEASE_EPONYM_SUFFIXES) + r")\b",
    re.I,
)
_DOSE_DUR_RE = re.compile(
    r"(?i)\b(?:\d+(?:\.\d+)?[ \t]*(?:mg/dL|mmol/L|mIU/L|g/dL|mg|mcg|ml|g|iu|puffs?|/uL|mmHg|/min|%|F|°F|yrs?|years?|months?|wks?|weeks?|days?|hrs?|hours?)|"
    r"[01]-[01]-[01]|(?:BD|OD|TDS|QID|SOS|IV|IM|PO)\b(?:[ \t]*\([A-Z]+\))?(?:[ \t]*x[ \t]*\d+[ \t]*(?:days?|wks?|weeks?|months?))?|"
    r"x[ \t]*\d+[ \t]*(?:days?|wks?|weeks?|months?|yrs?)|"
    r"BP[ \t]+\d{2,3}/\d{2,3}|Pulse[ \t]+\d{2,3}(?:/min)?|Temp[ \t]+\d{2,3}(?:\.\d+)?[ \t]*°?[FC]|SpO2[ \t]+\d{2,3}%?|"
    r"Ward[ \t]+\d+[ \t]*\([^)]+\)|Bed[ \t]+\d+|ASHA[ \t]+workers?)\b"
)
# Lab reference ranges only: never match 5-digit phone halves, dates (dd-mm-yyyy), ABHA, or IDs
_LAB_RANGE_RE = re.compile(
    r"(?<![A-Za-z-/\d.])(?!\d{2}-\d{2}-\d{4})(?![6-9]\d{4}-\d{5})\d{1,6}(?:\.\d+)?\s*-\s*\d{1,6}(?:\.\d+)?(?![A-Za-z-/\d.])"
)
_ICD_RE = re.compile(r"\b[A-TV-Z]\d{2}(?:\.\d{1,2})?\b")
_HIST_CLAUSE_RE = re.compile(r"(?i)\b(?:No\s+)?(?:K/C/O|H/O)\s*:?\s*[^.\n;]+")
_CO_CLAUSE_RE = re.compile(r"(?i)\bC/O\s*:?\s*[^.\n;]+")

# Base64 candidate runs
_B64_RE = re.compile(r"^[A-Za-z0-9+/]{24,}={0,2}$")
_UNICODE_ESCAPE_RE = re.compile(r"\\u([0-9a-fA-F]{4})")

# Dates
_ISO_DT_RE = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})(T\d{2}:\d{2}:\d{2}(?:[+-]\d{2}:\d{2}|Z)?)?\b")
_SLASH_DATE_RE = re.compile(r"\b(\d{1,2})([/.-])(\d{1,2})\2(\d{4})\b")
_MONTH_MAP = {
    "jan": 1, "january": 1, "feb": 2, "february": 2, "mar": 3, "march": 3,
    "apr": 4, "april": 4, "may": 5, "jun": 6, "june": 6, "jul": 7, "july": 7,
    "aug": 8, "august": 8, "sep": 9, "sept": 9, "september": 9,
    "oct": 10, "october": 10, "nov": 11, "november": 11, "dec": 12, "december": 12,
}
_MONTH_INV = {
    1: "January", 2: "February", 3: "March", 4: "April", 5: "May", 6: "June",
    7: "July", 8: "August", 9: "September", 10: "October", 11: "November", 12: "December",
}
_TEXT_DATE_RE = re.compile(
    r"\b(\d{1,2})\s+(Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)(?:\s+(\d{4}))?\b",
    re.I,
)

# Anchored Name/Relation Regexes (use [ \t] to avoid crossing line boundaries)
_NAME_LABEL_LATIN_RE = re.compile(
    r"(?i)(\b(?:Patient(?:'s)?[ \t]*Name|Patient(?=[ \t]*[:：])|Name|Pt\.?[ \t]*Name|Attendant|Guardian|Witnessed[ \t]+by|called[ \t]+by|Report[ \t]+of|Referral)[ \t]*[:：]?[ \t]*)"
    r"([A-Za-z\u0900-\u097f][A-Za-z\u0900-\u097f.'-]*(?:[ \t]+[A-Za-z\u0900-\u097f][A-Za-z\u0900-\u097f.'-]*){0,3}(?:[ \t]*\([^)\n]{2,50}\))?)"
)
_NAME_LABEL_DEVA_RE = re.compile(
    r"((?:मरीज़[ \t]+का[ \t]+नाम|रोगी[ \t]+का[ \t]+नाम|पति[ \t]+का[ \t]+नाम|पिता/पति[ \t]+का[ \t]+नाम|पिता[ \t]+का[ \t]+नाम|नाम)[ \t]*[:：][ \t]*)"
    r"([\u0900-\u097fA-Za-z]+(?:[ \t]+[\u0900-\u097fA-Za-z]+){0,3}(?:[ \t]*\([^)\n]{2,50}\))?)"
)
_DEVA_WORD_SEQ_RE = re.compile(r"[\u0900-\u097f]{2,}(?:[ \t]+[\u0900-\u097f]{2,}){0,3}")
_REL_ANCHOR_RE = re.compile(
    r"(?i)(\b(?:S/[oO]|D/[oO]|W/[oO]|C/[oO]|पुत्र|पुत्री|पत्नी)[ \t]*[:.]?[ \t]*)"
    r"([A-Za-z\u0900-\u097f][A-Za-z\u0900-\u097f.'-]*(?:[ \t]+[A-Za-z\u0900-\u097f][A-Za-z\u0900-\u097f.'-]*){0,2})"
)
_HONORIFIC_LATIN_RE = re.compile(
    r"\b([A-Z][a-z]+(?:[ \t]+[A-Z][a-z]+)?)[ \t]+(" + "|".join(sorted(HONORIFIC_SUFFIXES)) + r")\b"
)
_HINGLISH_POSSESSIVE_RE = re.compile(
    r"\b([A-Z][a-z]{2,}(?:[ \t]+[A-Z][a-z]{2,})?)(?='s[ \t]+attendant|[ \t]+(?:ka|ki|ke|ko|ne)[ \t]+(?:BP|condition|tabiyat|bukhar|sugar|fever|health|dawa|report))"
)
_DEVA_KO_RE = re.compile(
    r"([\u0900-\u097f]{2,}(?:[ \t]+[\u0900-\u097f]{2,}){0,2})[ \t]+(?:को|की|के|ने|जी|साहब|दीदी|बहनजी|भैया|अम्मा)\b"
)
_SAMPLE_FROM_RE = re.compile(r"(?i)(Sample[ \t]+received[ \t]+from[ \t]+)([A-Za-z]+(?:'s)?)")
_COMMA_REL_NAME_RE = re.compile(r"\b([A-Z][a-z]{2,}(?:[ \t]+[A-Z][a-z]{2,})?)[ \t]*,[ \t]*(?=(?:W/o|S/o|D/o|\d{1,3}[ \t]*(?:Y|Yrs?)[ \t]*/[ \t]*[MF]))")
_TIMELOG_ROLE_NAME_RE = re.compile(
    r"(?i)(?:(?:\b\d{1,2}:\d{2}[ \t]*[-–][ \t]*)|^)([A-Z][a-z]+(?:[ \t]+[A-Z][a-z]+){1,3})(?=[ \t]*\((?:attendant|visitor|patient|relative|guardian|caregiver|son|daughter|wife|husband))"
)

# Age in text
_AGE_LABEL_RE = re.compile(r"(?i)((?:\bAge(?:/Sex)?|आयु)\s*[:：]\s*)(\d{1,3})(\s*(?:Yrs?|Y|/|\s|वर्ष))")
_AGE_INLINE_RE = re.compile(r"\b(\d{1,3})\s*(Yrs?|Y)\s*/\s*([MF])\b")
_AGE_DEVA_RE = re.compile(r"\b(\d{1,3})\s*वर्ष\b")

_SYMPTOM_HINTS = {
    "fever", "pain", "cough", "cold", "weakness", "vomiting", "headache", "breathlessness",
    "chest", "abdomen", "bukhar", "dm", "dm2", "htn", "diarrhoea", "diarrhea", "rash",
    "swelling", "dizziness", "uti", "dengue", "typhoid", "malaria", "dyspnea", "nausea",
    "acute", "chronic", "gastritis", "asthma", "hypothyroidism", "hypertension", "diabetes",
    "pneumonia", "hepatitis", "anaemia", "anemia", "tuberculosis", "infection",
}
_ADDRESS_HINTS = {"h.no", "house", "makan", "ward", "mohalla", "vill", "tehsil", "distt", "district", "s/o", "d/o", "w/o"}


@dataclass
class ClueSet:
    """Extracted clues for a single patient-bearing record."""
    file_rel: str
    loc: str
    uhids: set[str] = field(default_factory=set)
    aadhaars: set[str] = field(default_factory=set)
    abhas: set[str] = field(default_factory=set)
    pans: set[str] = field(default_factory=set)
    insurances: set[str] = field(default_factory=set)
    phones: set[str] = field(default_factory=set)
    emails: set[str] = field(default_factory=set)
    name_raw: list[str] = field(default_factory=list)
    name_tokens: list[str] = field(default_factory=list)
    relative_raw: list[str] = field(default_factory=list)
    relative_tokens: list[str] = field(default_factory=list)
    dob: dt.date | None = None
    age: int | None = None
    gender: str | None = None  # "M", "F", "O"
    village: str | None = None
    district: str | None = None
    pin: str | None = None
    diagnoses: set[str] = field(default_factory=set)
    patient_key: str | None = None
    pseudonym: str | None = None
    date_shift: int = 0
    is_rare_combo: bool = False
    age_width: int = 5
    suppress_district: bool = False


class DetectionStats:
    def __init__(self) -> None:
        self.identifiers = 0
        self.records = 0
        self.linked = 0
        self.generalized = 0
        self.suppressed = 0


def classify_field(key: str) -> str:
    """Classify a CSV column header or JSON key semantically (language/variant agnostic)."""
    norm = normalize_text(key)
    if has_devanagari(norm):
        norm = devanagari_to_latin(norm)
    k = re.sub(r"[^a-z0-9]+", "_", norm.lower()).strip("_")
    if not k:
        return "other"
    toks = set(k.split("_"))

    if k in {"doctor", "ref_doctor", "referred_by", "consultant", "physician", "practitioner"}:
        return "doctor"
    if k in {"source_path", "filepath", "file_path", "export_path", "export_origin"}:
        return "source_path"
    if k in {"remarks", "notes", "note", "comment", "comments", "description", "div", "narrative", "tippanee", "tippani", "handover"} or "summary" in k:
        return "free_text"
    if any(t in k for t in ("aadhaar", "aadhar", "uidai")):
        return "aadhaar"
    if "abha" in k and ("addr" in k or "phr" in k):
        return "abha_address"
    if "abha" in k or "health_id" in k:
        return "abha"
    if k == "pan" or k.endswith("_pan") or k.startswith("pan_"):
        return "pan"
    if any(t in k for t in ("pmjay", "policy", "insurance")):
        return "insurance"
    if any(t in k for t in ("uhid", "mrn", "patient_ref", "patientid", "patient_id", "reg_no", "regno", "ipd", "adm_no", "aaeedee", "uid", "case_id")):
        return "uhid"
    if any(t in k for t in ("phone", "mobile", "mob", "callback", "telephone", "tel", "contact")):
        return "phone"
    if "email" in k or "e_mail" in k:
        return "email"
    if any(t in k for t in ("relative", "guardian", "caregiver", "attendant", "father", "husband", "spouse", "next_of_kin", "raised_by")) or bool(toks & {"pita", "pati"}):
        return "relative_name"
    if k in {"relation", "relationship"}:
        return "relation"
    if any(t in k for t in ("dob", "birth", "janm")):
        return "dob"
    if k in {"age", "umar", "umr", "ayu", "age_sex", "agesex", "ageyrs", "age_yrs"}:
        return "age"
    if k in {"gender", "sex", "ling"}:
        return "gender"
    if any(t in k for t in ("village", "gaon", "gram")):
        return "village"
    if k in {"tehsil", "taluka"}:
        return "tehsil"
    if any(t in k for t in ("district", "jila", "distt")):
        return "district"
    if any(t in k for t in ("pin", "zip", "postal")):
        return "pin"
    if any(t in k for t in ("address", "pata", "street", "house", "mohalla", "locality", "landmark")):
        return "address"
    if k in {"city"}:
        return "village"
    if k in {"first_name", "firstname", "first"}:
        return "first_name"
    if k in {"last_name", "lastname", "last"}:
        return "last_name"
    if any(t in k for t in ("patient_name", "patientname", "patients_name", "full_name", "beneficiary_name", "patient_label", "patientfullname")) or bool(toks & {"naam", "nam"}):
        return "patient_name"
    if k == "name":
        return "ambiguous_name"
    if any(t in k for t in ("date", "registered", "collected", "reported", "authoredon", "recordeddate", "effective", "performed", "timestamp", "lastupdated", "exported_at", "generated_at", "generated", "tithi", "admit", "disch")) or bool(toks & {"dt"}):
        return "date"
    if k in {"test", "result", "unit", "ref_range", "range", "loinc", "package_code", "procedure", "amount_inr",
             "status", "department", "category", "schedule", "form", "trip", "from", "to", "diagnosis",
             "clinicalstatus", "dosageinstruction", "medicationcodeableconcept", "valuequantity", "code",
             "section", "title", "intent", "system", "hospital", "lab", "facility", "report_type",
             "schema", "exported_by", "week", "wards", "visits", "bill_no", "visit_id", "sample_id", "id"}:
        return "clinical"
    return "other"


def norm_gender(val: str) -> str | None:
    s = normalize_text(val).strip().lower()
    if s in {"m", "male", "पुरुष"}:
        return "M"
    if s in {"f", "female", "महिला"}:
        return "F"
    if s in {"o", "other"}:
        return "O"
    return None


def norm_phone(val: str) -> str | None:
    d = "".join(ch for ch in fold_indic_digits(val) if ch.isdigit())
    if len(d) >= 10:
        last10 = d[-10:]
        if last10[0] in "6789":
            return last10
    return None


def parse_date_str(val: str, month_first: bool = False) -> tuple[dt.date | None, str | None]:
    """Parse ISO, slash, dash, or month-name date. Returns (date_obj, format_tag)."""
    s = fold_indic_digits(val.strip())
    m = _ISO_DT_RE.fullmatch(s)
    if m:
        try:
            return dt.date(int(m.group(1)), int(m.group(2)), int(m.group(3))), "iso_dt" if m.group(4) else "iso"
        except ValueError:
            return None, None
    m = _SLASH_DATE_RE.fullmatch(s)
    if m:
        a, sep, b, y = int(m.group(1)), m.group(2), int(m.group(3)), int(m.group(4))
        day, mon = (b, a) if month_first else (a, b)
        fmt_mode = "mdy" if month_first else "dmy"
        if mon > 12 and day <= 12:
            day, mon = mon, day
            fmt_mode = "dmy" if month_first else "mdy"
        try:
            return dt.date(y, mon, day), f"{fmt_mode}:{sep}"
        except ValueError:
            try:
                alt_mode = "dmy" if fmt_mode == "mdy" else "mdy"
                return dt.date(y, day, mon), f"{alt_mode}:{sep}"
            except ValueError:
                return None, None
    m = _TEXT_DATE_RE.fullmatch(s)
    if m:
        day = int(m.group(1))
        mon = _MONTH_MAP.get(m.group(2).lower(), 0)
        yr = int(m.group(3)) if m.group(3) else 2026
        try:
            return dt.date(yr, mon, day), "text_y" if m.group(3) else "text_noy"
        except ValueError:
            return None, None
    return None, None


def format_date_like(d: dt.date, fmt_tag: str, orig: str) -> str:
    if fmt_tag == "iso":
        return d.strftime("%Y-%m-%d")
    if fmt_tag == "iso_dt":
        suffix = orig[10:]
        return d.strftime("%Y-%m-%d") + suffix
    if fmt_tag.startswith("dmy:"):
        sep = fmt_tag.split(":", 1)[1]
        return f"{d.day:02d}{sep}{d.month:02d}{sep}{d.year:04d}"
    if fmt_tag.startswith("mdy:"):
        sep = fmt_tag.split(":", 1)[1]
        return f"{d.month:02d}{sep}{d.day:02d}{sep}{d.year:04d}"
    if fmt_tag == "text_y":
        return f"{d.day} {_MONTH_INV[d.month]} {d.year}"
    if fmt_tag == "text_noy":
        return f"{d.day} {_MONTH_INV[d.month]}"
    return d.strftime("%Y-%m-%d")


def is_clean_non_pii_word(word: str) -> bool:
    w = normalize_text(word).strip(".,'\"()[]{}:;!?-").lower()
    if not w:
        return True
    if w in NON_PII_WORDS:
        return True
    if has_devanagari(w) and w in NON_PII_WORDS:
        return True
    return False


def is_plausible_name_span(span: str, require_gazetteer: bool = False) -> bool:
    """Verify a candidate name span is not a clinical term, hospital, or place."""
    s = normalize_text(span).strip(".,'\"()[]{}:;!? ")
    if not s or len(s) < 2:
        return False
    low = s.lower()
    if any(hm in low for hm in HOSPITAL_MARKERS):
        return False
    if _EPONYM_RE.search(s):
        return False
    toks = [t.strip(".,'\"") for t in re.split(r"\s+", s) if t.strip(".,'\"")]
    if not toks:
        return False
    # Reject if all tokens are common non-PII/clinical words
    non_stop = [t for t in toks if not is_clean_non_pii_word(t) and t.lower() not in TITLE_PREFIXES]
    if not non_stop:
        return False
    if require_gazetteer:
        for t in non_stop:
            tl = t.lower()
            if tl in INDIAN_FIRST_NAMES or tl in INDIAN_SURNAMES or t in DEVANAGARI_NAMES:
                return True
        return False
    return True


def _decode_unicode_escapes(text: str) -> str:
    """Decode literal \\uXXXX escapes (e.g., hidden Devanagari) when present in text."""
    if "\\u" not in text:
        return text
    def _rep(m: re.Match[str]) -> str:
        cp = int(m.group(1), 16)
        # Only decode non-ASCII printable scripts (like Devanagari)
        if cp >= 0x0080:
            return chr(cp)
        return m.group(0)
    return _UNICODE_ESCAPE_RE.sub(_rep, text)


def extract_names_from_text(text: str, clues: ClueSet | None = None) -> list[str]:
    """Extract patient and relative names from free text for global Pass-2 lexicon + clues."""
    txt = _decode_unicode_escapes(text)
    found: list[str] = []

    # Shield doctors, eponyms, and K/C/O or H/O history clauses first
    masked = _DR_LATIN_RE.sub(" [DR] ", txt)
    masked = _DR_DEVA_RE.sub(" [DR] ", masked)
    masked = _EPONYM_RE.sub(" [EPONYM] ", masked)
    masked = _HIST_CLAUSE_RE.sub(" [HIST] ", masked)

    for rx in (_NAME_LABEL_LATIN_RE, _NAME_LABEL_DEVA_RE):
        for m in rx.finditer(masked):
            cand = m.group(2).strip(" ,.;:-")
            # Stop before inline labels like Age/Reg/Mob
            cand = re.split(r"\s+(?:Age|Sex|Reg|UHID|Mob|आयु|लिं|पंजी)", cand, 1, flags=re.I)[0].strip()
            # Split out any parenthesized transliteration/alias like "अनीता देवी (Anita Devi)"
            paren_m = re.search(r"\(([^)\n]+)\)", cand)
            parts_to_check = [re.sub(r"\s*\([^)\n]+\)", "", cand).strip()]
            if paren_m:
                parts_to_check.append(paren_m.group(1).strip())
            for part in parts_to_check:
                if is_plausible_name_span(part):
                    found.append(part)
                    if clues is not None and not any(x in m.group(1).lower() for x in ("attendant", "guardian", "पति", "पिता")):
                        if not clues.name_raw:
                            clues.name_raw.append(part)
                            clues.name_tokens = tokenize_name(part)

    for m in _REL_ANCHOR_RE.finditer(masked):
        prefix = m.group(1).lower()
        cand = m.group(2).strip(" ,.;:-")
        if prefix.startswith("c/o"):
            cand_low = cand.lower()
            first_tok = cand.split()[0] if cand.split() else ""
            if any(w in cand_low.split() for w in _SYMPTOM_HINTS) and not any(a in cand_low for a in _ADDRESS_HINTS):
                continue
            # Lowercase first word after C/O is a complaint unless in address/name list
            if first_tok and first_tok[0].islower() and first_tok not in INDIAN_FIRST_NAMES:
                continue
        if is_plausible_name_span(cand):
            found.append(cand)
            if clues is not None and not clues.relative_tokens:
                clues.relative_raw.append(cand)
                clues.relative_tokens = tokenize_name(cand)

    for rx in (_HONORIFIC_LATIN_RE, _HINGLISH_POSSESSIVE_RE, _DEVA_KO_RE, _COMMA_REL_NAME_RE, _TIMELOG_ROLE_NAME_RE):
        for m in rx.finditer(masked):
            cand = m.group(1).strip(" ,.;:-")
            if is_plausible_name_span(cand):
                found.append(cand)
                if clues is not None and not clues.name_raw:
                    clues.name_raw.append(cand)
                    clues.name_tokens = tokenize_name(cand)

    return found


def extract_clues_from_text(text: str, clues: ClueSet, month_first: bool = False) -> None:
    """Populate ClueSet from free text (prescriptions, discharge summaries, nursing notes)."""
    txt = _decode_unicode_escapes(text)
    for m in _MRN_RE.finditer(txt):
        clues.uhids.add(m.group(0).upper())
    for m in _MRN_LABELED_RE.finditer(txt):
        val = m.group(2).strip()
        if len(val) >= 4:
            clues.uhids.add(val.upper())
    for m in _ABHA_NUM_RE.finditer(txt):
        clues.abhas.add("".join(c for c in fold_indic_digits(m.group(0)) if c.isdigit()))
    txt_no_abha = _ABHA_NUM_RE.sub(" [ABHA] ", txt)
    for m in _AADHAAR_RE.finditer(txt_no_abha):
        clues.aadhaars.add("".join(c for c in fold_indic_digits(m.group(0)) if c.isdigit()))
    for m in _PAN_RE.finditer(txt):
        clues.pans.add(m.group(0).upper())
    for m in _EMAIL_RE.finditer(txt):
        clues.emails.add(m.group(0).lower())
    for m in _PHONE_RE.finditer(txt):
        p = norm_phone(m.group(0))
        if p:
            clues.phones.add(p)
    for m in _PHONE_LABELED_RE.finditer(txt):
        p = norm_phone(m.group(2))
        if p:
            clues.phones.add(p)

    extract_names_from_text(txt, clues=clues)

    # Age & Gender
    m_age = _AGE_LABEL_RE.search(txt) or _AGE_INLINE_RE.search(txt)
    if m_age:
        try:
            g_idx = 2 if m_age.re is _AGE_LABEL_RE else 1
            a = int(m_age.group(g_idx))
            if 0 <= a <= 120:
                clues.age = a
        except ValueError:
            pass
    elif (m_deva := _AGE_DEVA_RE.search(fold_indic_digits(txt))):
        a = int(m_deva.group(1))
        if 0 <= a <= 120:
            clues.age = a

    if re.search(r"\b\d{1,3}\s*(?:Yrs?|Y)?\s*/\s*F\b|महिला", txt, re.I):
        clues.gender = "F"
    elif re.search(r"\b\d{1,3}\s*(?:Yrs?|Y)?\s*/\s*M\b|पुरुष", txt, re.I):
        clues.gender = "M"
    elif re.search(r"\bW/o\b|पत्नी", txt, re.I):
        clues.gender = "F"
    elif re.search(r"\bS/o\b|पुत्र", txt, re.I):
        clues.gender = "M"

    # DOB
    m_dob = re.search(r"(?i)\bDOB\s*[:：]\s*([^\s,\n]+)", txt)
    if m_dob:
        d, _ = parse_date_str(m_dob.group(1), month_first=month_first)
        if d:
            clues.dob = d

    # Village / District / PIN
    m_v = _VILLAGE_ANCHORED_RE.search(txt)
    if m_v:
        v_raw = re.sub(r"(?i)^(?:vill\.?|village|gaon|ग्राम|गाँव)\s*[:.]?\s*", "", m_v.group(0)).strip()
        if v_raw:
            clues.village = " ".join(tokenize_name(v_raw))
    m_p = _PIN_LABELED_RE.search(txt)
    if m_p:
        clues.pin = m_p.group(2)
    m_d = re.search(r"(?i)(?:\bDistt\.?|\bDistrict(?!\s+Hospital\b)|जिला)[ \t]*[:.]?[ \t]*([A-Za-z\u0900-\u097f]+(?:[ \t]+[A-Za-z\u0900-\u097f]+)?)", txt)
    if m_d:
        clues.district = " ".join(tokenize_name(m_d.group(1)))

    # ICD codes for rare-condition tracking
    for m in _ICD_RE.finditer(txt):
        clues.diagnoses.add(m.group(0).upper())


def filename_has_pii(rel: str, known_name_tokens: set[str]) -> bool:
    """Detect if a filename/path component contains PII (phone, UHID, Aadhaar, or person names)."""
    stem = rel.rsplit("/", 1)[-1].rsplit(".", 1)[0]
    stem_spaced = stem.replace("_", " ")
    if _PHONE_RE.search(stem_spaced) or _AADHAAR_RE.search(stem_spaced) or _MRN_RE.search(stem_spaced):
        return True
    toks = [t for t in re.split(r"[_\-\s.]+", stem) if t]
    for i, tok in enumerate(toks):
        if tok.isdigit() and len(tok) == 10 and tok[0] in "6789":
            return True
        tl = tok.lower()
        if tl in known_name_tokens and not is_clean_non_pii_word(tok):
            return True
        if i + 1 < len(toks):
            nxt = toks[i + 1].lower()
            if tl in INDIAN_FIRST_NAMES and nxt in INDIAN_SURNAMES:
                return True
    return False


def sanitize_filename(rel: str, known_name_tokens: set[str], used_paths: set[str]) -> str:
    """Return original relative path if clean, or deterministically rename any PII-bearing dir/file segments."""
    segments = rel.split("/")
    new_segments: list[str] = []
    changed = False
    for idx_seg, seg in enumerate(segments):
        is_last = idx_seg == len(segments) - 1
        if filename_has_pii(seg, known_name_tokens):
            changed = True
            if is_last:
                ext = "." + seg.rsplit(".", 1)[1] if "." in seg else ""
                tok = hmac_token("fname", rel, length=8)
                new_segments.append(f"file_{tok}{ext}")
            else:
                tok = hmac_token("dname", "/".join(segments[: idx_seg + 1]), length=8)
                new_segments.append(f"dir_{tok}")
        else:
            new_segments.append(seg)

    candidate = "/".join(new_segments)
    if not changed:
        used_paths.add(candidate)
        return candidate
    idx = 1
    base_cand = candidate
    while candidate in used_paths:
        if "." in base_cand.rsplit("/", 1)[-1]:
            stem, ext = base_cand.rsplit(".", 1)
            candidate = f"{stem}_{idx}.{ext}"
        else:
            candidate = f"{base_cand}_{idx}"
        idx += 1
    used_paths.add(candidate)
    return candidate


# --------------------------------------------------------------------------- Text Scrubber
class GlobalScrubber:
    """Dataset-aware deterministic scrubber applying clinical protection + Pass-2 name lexicon."""

    def __init__(self, stats: DetectionStats, age_width: int = 5, date_shift_max: int = 180) -> None:
        self.stats = stats
        self.age_width = age_width
        self.date_shift_max = date_shift_max
        self.known_full_names: list[str] = []
        self.known_first_tokens: set[str] = set()
        self.known_phonetic_tokens: set[str] = set()
        self._compiled_full_rx: re.Pattern[str] | None = None
        self._compiled_token_rx: re.Pattern[str] | None = None

    def register_names(self, names: list[str]) -> None:
        for n in names:
            s = normalize_text(n).strip(".,'\"()[]{}:;!? ")
            if not s or len(s) < 2 or not is_plausible_name_span(s):
                continue
            cleaned_parts = [
                tok for tok in re.split(r"[\s,]+", s)
                if tok.strip(".,'\"-").lower() not in HONORIFIC_SUFFIXES
                and tok.strip(".,'\"-").lower() not in TITLE_PREFIXES
                and not is_clean_non_pii_word(tok)
            ]
            if not cleaned_parts:
                continue
            self.known_full_names.append(" ".join(cleaned_parts))
            for tok in cleaned_parts:
                tl = tok.strip(".,'\"-").lower()
                if len(tl) >= 3:
                    self.known_first_tokens.add(tl)
                    pk = phonetic_key(tl)
                    if len(pk) >= 3:
                        self.known_phonetic_tokens.add(pk)
                if has_devanagari(tok) and len(tok) >= 2 and tok not in NON_PII_WORDS:
                    self.known_first_tokens.add(tok)

    def finalize_lexicon(self) -> None:
        # Sort longest-first deterministically for regex alternation
        uniq_full = sorted(set(self.known_full_names), key=lambda x: (-len(x), x))
        if uniq_full:
            escaped = [re.escape(x) for x in uniq_full if len(x) >= 3]
            if escaped:
                # Alphanumeric-boundary matching (handles underscores in keys/paths too)
                parts = []
                for esc, raw in zip(escaped, uniq_full):
                    if raw[0].isascii() and raw[0].isalnum():
                        esc = r"(?<![A-Za-z0-9])" + esc
                    if raw[-1].isascii() and raw[-1].isalnum():
                        esc = esc + r"(?![A-Za-z0-9])"
                    parts.append(esc)
                self._compiled_full_rx = re.compile("|".join(parts), re.I)

        toks = sorted(
            {t for t in self.known_first_tokens if len(t) >= 4 or has_devanagari(t)},
            key=lambda x: (-len(x), x),
        )
        if toks:
            parts = []
            for t in toks:
                esc = re.escape(t)
                if t.isascii():
                    esc = r"(?<![A-Za-z0-9])" + esc + r"(?![A-Za-z0-9])"
                parts.append(esc)
            self._compiled_token_rx = re.compile("|".join(parts), re.I)

    def _shield_clinical(self, text: str) -> tuple[str, dict[str, str]]:
        shields: dict[str, str] = {}
        counter = 0

        def _put(m: re.Match[str]) -> str:
            nonlocal counter
            key = f"\x00SHLD{counter}\x00"
            counter += 1
            shields[key] = m.group(0)
            return key

        def _maybe_co(m: re.Match[str]) -> str:
            clause = m.group(0)
            body = re.sub(r"(?i)^C/O\s*:?\s*", "", clause).strip()
            low = body.lower()
            # If body looks like an address or relative, do NOT shield
            if any(h in low for h in _ADDRESS_HINTS):
                return clause
            if any(w in low.split() for w in _SYMPTOM_HINTS):
                return _put(m)
            first_word = body.split(",")[0].split()[0] if body.split() else ""
            fl = first_word.lower()
            if fl in self.known_first_tokens or fl in INDIAN_FIRST_NAMES:
                return clause
            if first_word and first_word[0].islower():
                return _put(m)
            return _put(m)

        s = _DR_LATIN_RE.sub(_put, text)
        s = _DR_DEVA_RE.sub(_put, s)
        s = _EPONYM_RE.sub(_put, s)
        s = _HIST_CLAUSE_RE.sub(_put, s)
        s = _CO_CLAUSE_RE.sub(_maybe_co, s)
        s = _DOSE_DUR_RE.sub(_put, s)
        s = _LAB_RANGE_RE.sub(_put, s)
        s = _ICD_RE.sub(_put, s)
        return s, shields

    def _unshield(self, text: str, shields: dict[str, str]) -> str:
        # Restore iteratively in case shields nested
        for _ in range(3):
            changed = False
            for k, v in shields.items():
                if k in text:
                    text = text.replace(k, v)
                    changed = True
            if not changed:
                break
        return text

    def scrub_b64_if_present(self, text: str) -> str:
        s = text.strip()
        if not _B64_RE.fullmatch(s):
            return text
        try:
            decoded = base64.b64decode(s + "=" * (-len(s) % 4), validate=False).decode("utf-8", errors="strict")
        except (binascii.Error, UnicodeDecodeError, ValueError):
            return text
        # Verify decoded text is mostly printable text
        if not decoded or sum(1 for c in decoded if c.isprintable() or c in "\r\n\t") / len(decoded) < 0.9:
            return text
        cleaned = self.scrub_text(decoded)
        return base64.b64encode(cleaned.encode("utf-8")).decode("ascii")

    def scrub_text(
        self,
        text: str,
        shift_days: int = 0,
        month_first: bool = False,
        age_width: int | None = None,
        suppress_location: bool = False,
    ) -> str:
        """Scrub PII in arbitrary text while preserving clinical traps and shifting dates/ages."""
        if not text:
            return text
        # Handle pure base64 payload strings
        if _B64_RE.fullmatch(text.strip()):
            return self.scrub_b64_if_present(text)

        # Decode URL-encoded (%XX) payloads when present (e.g., %40 email or %20/%3A text)
        if "%" in text and (len(_URL_PCT_RE.findall(text)) >= 2 or "%40" in text):
            unquoted = urllib.parse.unquote(text)
            if unquoted != text and sum(1 for c in unquoted if c.isprintable() or c in "\r\n\t") / max(1, len(unquoted)) >= 0.9:
                text = unquoted

        s = normalize_text(_decode_unicode_escapes(text))
        s, shields = self._shield_clinical(s)

        def _sub(rx: re.Pattern[str], repl: Any, txt: str) -> str:
            out, n = rx.subn(repl, txt)
            self.stats.identifiers += n
            return out

        # 1. Direct structured IDs & contact patterns
        s = _sub(_ABHA_ADDR_RE, "[ABHA_ADDRESS]", s)
        s = _sub(_EMAIL_RE, "[EMAIL]", s)
        s = _sub(_ABHA_NUM_RE, "[ABHA_NUMBER]", s)
        s = _sub(_AADHAAR_RE, "[AADHAAR]", s)
        s = _sub(_AADHAAR_MASKED_RE, "[AADHAAR_MASKED]", s)
        s = _sub(_PAN_RE, "[PAN]", s)
        s = _sub(_INSURANCE_RE, "[INSURANCE_ID]", s)
        s = _sub(_MRN_RE, "[MRN]", s)
        s = _sub(_MRN_LABELED_RE, r"\1[MRN]", s)
        s = _sub(_PHONE_RE, "[PHONE]", s)
        s = _sub(_PHONE_MASKED_RE, "[PHONE]", s)
        s = _sub(_PHONE_LABELED_RE, r"\1[PHONE]", s)

        # 1b. Windows export paths (e.g., D:\Exports\Place\410401\out.json)
        s = _sub(_WIN_PATH_RE, "[SOURCE_PATH]", s)

        # 2. Addresses, houses, localities, landmarks, villages, PINs
        s = _sub(_HOUSE_RE, "[HOUSE]", s)
        s = _sub(_MOHALLA_RE, "[LOCALITY]", s)
        s = _sub(_RES_WARD_RE, "[LOCALITY]", s)
        s = _sub(_LANDMARK_RE, "[LANDMARK]", s)
        s = _sub(_VILLAGE_ANCHORED_RE, "[VILLAGE]", s)
        s = _ADDR_VILLAGE_BEFORE_TEHSIL_RE.sub(lambda m: f"{m.group(1)}[VILLAGE]" if not is_clean_non_pii_word(m.group(2)) else m.group(0), s)
        s = _sub(_TEHSIL_ANCHORED_RE, "[TEHSIL]", s)
        s = _PIN_LABELED_RE.sub(
            lambda m: m.group(1) + ("******" if suppress_location else generalize_pin(fold_indic_digits(m.group(2)), 3)),
            s,
        )

        # 3. Anchored names & relatives
        def _rep_label(m: re.Match[str]) -> str:
            prefix, val = m.group(1), m.group(2)
            # Split off trailing non-name labels like Age/Reg/Mob
            split_m = re.search(r"\s+(?:Age|Sex|Reg|UHID|Mob|आयु|लिं|पंजी)", val, re.I)
            tail = ""
            if split_m:
                tail = val[split_m.start():]
                val = val[:split_m.start()]
            # Check primary part (outside parens) and optional parenthesized alias
            main_part = re.sub(r"[ \t]*\([^)\n]+\)", "", val).strip()
            paren_m = re.search(r"\(([^)\n]+)\)", val)
            ok_main = is_plausible_name_span(main_part)
            ok_paren = paren_m is not None and is_plausible_name_span(paren_m.group(1))
            if ok_main or ok_paren:
                self.stats.identifiers += 1
                tag = "[RELATIVE_NAME]" if any(x in prefix.lower() for x in ("attendant", "guardian", "पति", "पिता")) else "[PATIENT_NAME]"
                if ok_main and ok_paren:
                    return f"{prefix}{tag} ({tag}){tail}"
                return f"{prefix}{tag}{tail}"
            return m.group(0)

        s = _NAME_LABEL_LATIN_RE.sub(_rep_label, s)
        s = _NAME_LABEL_DEVA_RE.sub(_rep_label, s)

        def _rep_rel(m: re.Match[str]) -> str:
            prefix, val = m.group(1), m.group(2)
            if prefix.lower().startswith("c/o"):
                low = val.lower()
                if any(w in low.split() for w in _SYMPTOM_HINTS) and not any(a in low for a in _ADDRESS_HINTS):
                    return m.group(0)
            if is_plausible_name_span(val):
                self.stats.identifiers += 1
                return f"{prefix}[RELATIVE_NAME]"
            return m.group(0)

        s = _REL_ANCHOR_RE.sub(_rep_rel, s)
        s = _SAMPLE_FROM_RE.sub(lambda m: f"{m.group(1)}[PATIENT_NAME]" if is_plausible_name_span(m.group(2)) else m.group(0), s)
        s = _TIMELOG_ROLE_NAME_RE.sub(lambda m: m.group(0).replace(m.group(1), "[PATIENT_NAME]", 1) if is_plausible_name_span(m.group(1)) else m.group(0), s)
        s = _HONORIFIC_LATIN_RE.sub(lambda m: f"[PATIENT_NAME] {m.group(2)}" if is_plausible_name_span(m.group(1)) else m.group(0), s)
        s = _HINGLISH_POSSESSIVE_RE.sub(lambda m: "[PATIENT_NAME]" if is_plausible_name_span(m.group(1)) else m.group(0), s)
        s = _COMMA_REL_NAME_RE.sub(lambda m: "[PATIENT_NAME], " if is_plausible_name_span(m.group(1)) else m.group(0), s)
        s = _DEVA_KO_RE.sub(
            lambda m: m.group(0).replace(m.group(1), "[PATIENT_NAME]", 1) if is_plausible_name_span(m.group(1)) else m.group(0),
            s,
        )

        # 4. Pass-2 global name lexicon + gazetteer sweep
        if self._compiled_full_rx is not None:
            def _rep_full(m: re.Match[str]) -> str:
                tok = m.group(0)
                if is_clean_non_pii_word(tok):
                    return tok
                self.stats.identifiers += 1
                return "[PATIENT_NAME]"
            s = self._compiled_full_rx.sub(_rep_full, s)

        if self._compiled_token_rx is not None:
            def _rep_tok(m: re.Match[str]) -> str:
                tok = m.group(0)
                if is_clean_non_pii_word(tok):
                    return tok
                self.stats.identifiers += 1
                return "[PATIENT_NAME]"
            s = self._compiled_token_rx.sub(_rep_tok, s)

        # Secondary Devanagari gazetteer sweep for unanchored Devanagari person names
        def _rep_deva_seq(m: re.Match[str]) -> str:
            seq = m.group(0)
            words = seq.split()
            if all(w in DEVANAGARI_NAMES and w not in NON_PII_WORDS for w in words):
                self.stats.identifiers += 1
                return "[PATIENT_NAME]"
            return seq
        s = _DEVA_WORD_SEQ_RE.sub(_rep_deva_seq, s)

        # 5. Shift dates & generalize ages in text
        if shift_days != 0:
            def _shift_iso(m: re.Match[str]) -> str:
                try:
                    d = dt.date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
                except ValueError:
                    return m.group(0)
                sd = d + dt.timedelta(days=shift_days)
                return sd.strftime("%Y-%m-%d") + (m.group(4) or "")

            def _shift_slash(m: re.Match[str]) -> str:
                d, fmt = parse_date_str(m.group(0), month_first=month_first)
                if d is None or fmt is None:
                    return m.group(0)
                return format_date_like(d + dt.timedelta(days=shift_days), fmt, m.group(0))

            def _shift_text_date(m: re.Match[str]) -> str:
                d, fmt = parse_date_str(m.group(0), month_first=month_first)
                if d is None or fmt is None:
                    return m.group(0)
                return format_date_like(d + dt.timedelta(days=shift_days), fmt, m.group(0))

            s = _ISO_DT_RE.sub(_shift_iso, s)
            s = _SLASH_DATE_RE.sub(_shift_slash, s)
            s = _TEXT_DATE_RE.sub(_shift_text_date, s)

        w = age_width if age_width is not None else self.age_width
        s = _AGE_LABEL_RE.sub(lambda m: f"{m.group(1)}{age_band(int(m.group(2)), w)}{m.group(3)}", s)
        s = _AGE_INLINE_RE.sub(lambda m: f"{age_band(int(m.group(1)), w)} {m.group(2)}/{m.group(3)}", s)
        s = _AGE_DEVA_RE.sub(lambda m: f"{age_band(int(m.group(1)), w)} वर्ष", s)

        # Restore protected clinical spans
        s = self._unshield(s, shields)
        return s

    def scrub_key(self, key: str) -> str:
        """Scrub PII from a JSON key or CSV header ONLY if it contains PII."""
        # Fast check: keep normal schema keys intact
        if key in {"patient", "patient_name", "patient_ref", "uhid", "name", "first", "last", "phone",
                   "mobile", "email", "address", "dob", "age", "gender", "sex", "village", "district",
                   "tehsil", "pin", "by_uhid", "by_name", "visits", "patients", "results", "tickets"}:
            return key
        cleaned = self.scrub_text(key)
        return cleaned
