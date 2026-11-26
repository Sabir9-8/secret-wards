"""Policy loader and HMAC-SHA256 keyed pseudonym/date-shift primitives."""
from __future__ import annotations

import datetime as dt
import hashlib
import hmac
import os
from pathlib import Path

_ALPHABET = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"  # Crockford base32: no I, L, O, U


def load_policy(path: Path) -> dict[str, int | str]:
    """Read key: value pairs from config/policy.yaml without requiring PyYAML."""
    policy: dict[str, int | str] = {}
    if path.exists():
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.split("#", 1)[0].strip()
            if ":" in line:
                k, v = (x.strip() for x in line.split(":", 1))
                if v.lstrip("-").isdigit():
                    policy[k] = int(v)
                else:
                    policy[k] = v
    return policy


def get_secret() -> bytes:
    """Read RAKSHA_SECRET from environment dynamically (never log or write)."""
    return os.environ.get("RAKSHA_SECRET", "").encode("utf-8")


def pseudonym(patient_key: str, secret: bytes | None = None) -> str:
    """Deterministic, non-reversible HMAC-SHA256 pseudonym (PSN-XXXXXX)."""
    sec = get_secret() if secret is None else secret
    digest = hmac.new(sec, patient_key.encode("utf-8"), hashlib.sha256).digest()
    n = int.from_bytes(digest[:5], "big")
    code = "".join(_ALPHABET[(n >> (5 * i)) & 31] for i in range(6))
    return f"PSN-{code}"


def hmac_token(namespace: str, value: str, length: int = 8, secret: bytes | None = None) -> str:
    """Deterministic keyed short token for safe filenames or anonymized IDs."""
    sec = get_secret() if secret is None else secret
    digest = hmac.new(sec, f"{namespace}|{value}".encode("utf-8"), hashlib.sha256).digest()
    n = int.from_bytes(digest[:8], "big")
    return "".join(_ALPHABET[(n >> (5 * i)) & 31] for i in range(length))


def date_shift_days(patient_key: str, max_days: int = 180, secret: bytes | None = None) -> int:
    """Per-patient shift in [-max_days, +max_days], never 0."""
    sec = get_secret() if secret is None else secret
    d = hmac.new(sec, b"shift|" + patient_key.encode("utf-8"), hashlib.sha256).digest()
    span = max(2, 2 * max_days)
    v = int.from_bytes(d[:4], "big") % span - max_days
    return v if v != 0 else 1


def shift_date(value: dt.date, patient_key: str, max_days: int = 180, secret: bytes | None = None) -> dt.date:
    """Shift a date by the patient's deterministic secret offset."""
    return value + dt.timedelta(days=date_shift_days(patient_key, max_days=max_days, secret=secret))


def age_band(age: int, width: int = 5) -> str:
    """Return a truthful age band containing `age`."""
    if age < 0:
        age = 0
    w = max(1, width)
    lo = (age // w) * w
    return f"{lo}-{lo + w - 1}"


def generalize_pin(pin: str, keep_digits: int = 3) -> str:
    """Generalize a 6-digit Indian PIN code to its prefix (e.g. 229***)."""
    digits = "".join(ch for ch in pin if ch.isdigit())
    if len(digits) != 6:
        return "[PIN]"
    if keep_digits <= 0:
        return "******"
    return digits[:keep_digits] + "*" * (6 - keep_digits)
