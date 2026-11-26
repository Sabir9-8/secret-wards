"""Safe multi-encoding ingestion, mislabeled/corrupt file quarantine, and format parsing."""
from __future__ import annotations

import csv
import io
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SUPPORTED_EXTS = {".csv", ".tsv", ".json", ".txt"}

# Bytes invalid in cp1252 or strong indicators of corrupt UTF-8/binary contamination
_CP1252_UNDEFINED = {0x81, 0x8D, 0x8F, 0x90, 0x9D}


@dataclass
class ParsedFile:
    rel: str
    ext: str
    sha256: str
    kind: str  # "csv", "json", "fhir", "txt"
    delimiter: str = ","
    rows: list[list[str]] | None = None
    json_data: Any = None
    text: str | None = None


def decode_bytes(raw: bytes) -> tuple[str | None, str | None]:
    """Decode UTF-8-BOM, UTF-16, UTF-8, or clean cp1252. Return (text, error_reason)."""
    if not raw.strip():
        return None, "empty_file"

    if raw.startswith(b"\xef\xbb\xbf"):
        try:
            text = raw[3:].decode("utf-8", errors="strict")
            if "\x00" in text:
                return None, "corrupt_encoding"
            return text, None
        except UnicodeDecodeError:
            return None, "corrupt_encoding"

    if raw.startswith((b"\xff\xfe", b"\xfe\xff")):
        try:
            text = raw.decode("utf-16", errors="strict")
            if "\x00" in text or "\ufffd" in text:
                return None, "corrupt_encoding"
            return text, None
        except UnicodeDecodeError:
            return None, "corrupt_encoding"

    # Reject UTF-16 BOM markers embedded mid-stream in non-UTF-16 files
    if b"\xff\xfe" in raw or b"\xfe\xff" in raw:
        return None, "corrupt_encoding"

    try:
        text = raw.decode("utf-8", errors="strict")
        if "\x00" in text:
            return None, "corrupt_encoding"
        return text, None
    except UnicodeDecodeError:
        pass

    # Strict cp1252 fallback: reject undefined bytes, control bytes, and Latin-1 mojibake
    for b in raw:
        if b in _CP1252_UNDEFINED:
            return None, "corrupt_encoding"
        if b < 0x20 and b not in (9, 10, 13):
            return None, "corrupt_encoding"
        if b >= 0xC0:
            # High bytes >= 0xC0 in non-UTF-8 text typically indicate truncated/corrupt UTF-8
            return None, "corrupt_encoding"

    try:
        return raw.decode("cp1252", errors="strict"), None
    except UnicodeDecodeError:
        return None, "undecodable_text"


def _looks_like_json_or_xml(text: str) -> bool:
    s = text.lstrip()
    if not s:
        return False
    if s.startswith(("{", "[")):
        try:
            obj = json.loads(s)
            return isinstance(obj, (dict, list))
        except (ValueError, RecursionError):
            return False
    if s.startswith("<") and ("</" in s or "/>" in s):
        return True
    return False


def ingest_file(path: Path, inp_root: Path, sha256_hex: str) -> tuple[ParsedFile | None, str | None]:
    """Read, validate, and parse a single input file, or return a quarantine reason."""
    rel = path.relative_to(inp_root).as_posix()
    ext = path.suffix.lower()
    if ext not in SUPPORTED_EXTS:
        return None, "unsupported_format"

    raw = path.read_bytes()
    text, err = decode_bytes(raw)
    if err is not None or text is None:
        return None, err or "undecodable_text"

    if ext in (".csv", ".tsv"):
        if _looks_like_json_or_xml(text):
            return None, "mislabeled_format"
        delim = "\t" if ext == ".tsv" else ","
        try:
            rows = list(csv.reader(io.StringIO(text, newline=""), delimiter=delim))
        except csv.Error:
            return None, "csv_parse_error"
        if not rows:
            return None, "empty_file"
        return ParsedFile(rel=rel, ext=ext, sha256=sha256_hex, kind="csv", delimiter=delim, rows=rows), None

    if ext == ".json":
        sys.setrecursionlimit(5000)
        try:
            data = json.loads(text)
        except (ValueError, RecursionError) as exc:
            return None, type(exc).__name__
        kind = "fhir" if isinstance(data, dict) and data.get("resourceType") == "Bundle" else "json"
        return ParsedFile(rel=rel, ext=ext, sha256=sha256_hex, kind=kind, json_data=data), None

    # .txt
    normalized_txt = text.replace("\r\n", "\n").replace("\r", "\n")
    return ParsedFile(rel=rel, ext=ext, sha256=sha256_hex, kind="txt", text=normalized_txt), None
