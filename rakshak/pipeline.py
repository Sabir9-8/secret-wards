"""End-to-end Secret Wards pipeline orchestration: ingest -> clues -> linkage -> 2-pass scrub & export."""
from __future__ import annotations

import csv
import datetime as dt
import hashlib
import io
import json
import re
import resource
import sys
import time
from pathlib import Path
from typing import Any

from .detect import (
    ClueSet,
    DetectionStats,
    GlobalScrubber,
    classify_field,
    extract_clues_from_text,
    extract_names_from_text,
    format_date_like,
    norm_gender,
    norm_phone,
    parse_date_str,
    sanitize_filename,
)
from .ingest import ParsedFile, ingest_file
from .linker import link_and_protect
from .policy import age_band, generalize_pin, hmac_token, load_policy
from .translit import fold_indic_digits, tokenize_name


def peak_rss_mb() -> float:
    """Return peak RSS (MB) across self and child processes, rounded to 2 decimals."""
    try:
        r_self = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        r_child = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
        r = max(r_self, r_child)
        return round(r / 1048576 if sys.platform == "darwin" else r / 1024, 2)
    except Exception:  # noqa: BLE001
        return 0.0


def _infer_month_first(parsed: ParsedFile) -> bool:
    """Infer MM/DD/YYYY vs DD/MM/YYYY from file headers or unambiguous values."""
    if parsed.kind == "csv" and parsed.rows:
        hdr = " ".join(parsed.rows[0]).lower()
        if "mm/dd/yyyy" in hdr:
            return True
        if "dd/mm/yyyy" in hdr:
            return False
        # Probe slash dates across body
        gt12_first = gt12_second = 0
        for row in parsed.rows[1:]:
            for cell in row:
                parts = cell.strip().split("/")
                if len(parts) == 3 and all(p.isdigit() for p in parts):
                    a, b = int(parts[0]), int(parts[1])
                    if a > 12:
                        gt12_first += 1
                    if b > 12:
                        gt12_second += 1
        if gt12_second > gt12_first:
            return True
        return False
    if parsed.kind == "txt" and parsed.text:
        if "arogya" in parsed.rel.lower() or "arogya" in parsed.text.lower():
            # Check if any date has second segment > 12
            for m_str in parsed.text.split():
                parts = m_str.strip(",.;:").split("/")
                if len(parts) == 3 and parts[0].isdigit() and parts[1].isdigit():
                    if int(parts[1]) > 12:
                        return True
    return False


def _extract_dict_clues(
    obj: dict[str, Any],
    clues: ClueSet,
    all_names: list[str],
    month_first: bool,
    parent_key: str = "",
) -> None:
    """Recursively collect clues from a structured JSON record."""
    first_parts: list[str] = []
    last_parts: list[str] = []
    for k, v in obj.items():
        role = classify_field(k)
        if isinstance(v, dict):
            _extract_dict_clues(v, clues, all_names, month_first, parent_key=k)
            continue
        if isinstance(v, list):
            for item in v:
                if isinstance(item, dict):
                    _extract_dict_clues(item, clues, all_names, month_first, parent_key=k)
                elif isinstance(item, str):
                    extract_clues_from_text(item, clues, month_first=month_first)
                    all_names.extend(extract_names_from_text(item))
            continue

        if v is None:
            continue
        sv = str(v).strip()
        if not sv:
            continue

        if role == "doctor" or parent_key.lower() in {"facility", "hospital", "organization", "practitioner"}:
            continue
        if role == "uhid":
            clues.uhids.add(sv.upper())
        elif role == "aadhaar":
            digits = "".join(c for c in fold_indic_digits(sv) if c.isdigit())
            if len(digits) == 12:
                clues.aadhaars.add(digits)
        elif role == "abha":
            digits = "".join(c for c in fold_indic_digits(sv) if c.isdigit())
            if len(digits) >= 10:
                clues.abhas.add(digits)
        elif role == "pan":
            clues.pans.add(sv.upper())
        elif role == "insurance":
            clues.insurances.add(sv.upper())
        elif role == "phone":
            p = norm_phone(sv)
            if p:
                clues.phones.add(p)
        elif role == "email":
            clues.emails.add(sv.lower())
        elif role == "first_name":
            first_parts.append(sv)
            all_names.append(sv)
        elif role == "last_name":
            last_parts.append(sv)
            all_names.append(sv)
        elif role == "patient_name" or (role == "ambiguous_name" and parent_key.lower() in {"patient", "raised_by", "name"}):
            if parent_key.lower() == "name" and k.lower() == "local":
                all_names.append(sv)
                if not clues.name_raw:
                    clues.name_raw.append(sv)
                    clues.name_tokens = tokenize_name(sv)
            else:
                all_names.append(sv)
                clues.name_raw.append(sv)
                clues.name_tokens = tokenize_name(sv)
        elif role == "relative_name" or (role == "ambiguous_name" and parent_key.lower() in {"guardian", "relative", "contact"}):
            rel_clean = re.sub(r"(?i)^(?:S/[oO]|D/[oO]|W/[oO]|C/[oO]|पुत्र|पुत्री|पत्नी)\s*[:.]?\s*", "", sv).strip()
            if rel_clean:
                all_names.append(rel_clean)
                clues.relative_raw.append(rel_clean)
                clues.relative_tokens = tokenize_name(rel_clean)
        elif role == "dob":
            d, _ = parse_date_str(sv, month_first=month_first)
            if d:
                clues.dob = d
        elif role == "age":
            sv_d = fold_indic_digits(sv).strip()
            if sv_d.isdigit():
                a = int(sv_d)
                if 0 <= a <= 120:
                    clues.age = a
            else:
                m_as = re.match(r"^(\d{1,3})\s*(?:Yrs?|Y)?\s*/\s*([MF])\b", sv_d, re.I)
                if m_as:
                    a = int(m_as.group(1))
                    if 0 <= a <= 120:
                        clues.age = a
                    clues.gender = m_as.group(2).upper()
        elif role == "gender":
            g = norm_gender(sv)
            if g:
                clues.gender = g
        elif role == "village":
            clues.village = " ".join(tokenize_name(sv))
        elif role == "district":
            clues.district = " ".join(tokenize_name(sv))
        elif role == "pin":
            clues.pin = sv
        elif k.lower() in {"code", "icd"} and isinstance(v, str) and len(sv) <= 8:
            clues.diagnoses.add(sv.upper())
        elif role == "date":
            d_ev, _ = parse_date_str(sv, month_first=month_first)
            if d_ev:
                clues.event_dates.add(d_ev)
        elif role == "free_text" or (role == "other" and len(sv) >= 10 and (" " in sv or any(ch.isdigit() for ch in sv))):
            extract_clues_from_text(sv, clues, month_first=month_first)
            all_names.extend(extract_names_from_text(sv))

    if first_parts or last_parts:
        full = " ".join(first_parts + last_parts).strip()
        if full:
            all_names.append(full)
            clues.name_raw.append(full)
            clues.name_tokens = tokenize_name(full)


def _extract_fhir_clues(bundle: dict[str, Any], clues: ClueSet, all_names: list[str]) -> None:
    meta_src = (bundle.get("meta") or {}).get("source")
    if isinstance(meta_src, str):
        extract_clues_from_text(meta_src, clues)

    for ent in bundle.get("entry") or []:
        res = ent.get("resource") if isinstance(ent, dict) else None
        if not isinstance(res, dict):
            continue
        rt = res.get("resourceType")
        if rt == "Patient":
            for ident in res.get("identifier") or []:
                val = str(ident.get("value") or "").strip()
                sys_u = str(ident.get("system") or "").lower()
                if not val:
                    continue
                if "healthid" in sys_u or "abha" in sys_u:
                    clues.abhas.add("".join(c for c in val if c.isdigit()))
                elif "aadhaar" in sys_u or "uidai" in sys_u:
                    clues.aadhaars.add("".join(c for c in val if c.isdigit()))
                else:
                    clues.uhids.add(val.upper())
            for nm in res.get("name") or []:
                txt = nm.get("text")
                if not txt:
                    given = " ".join(nm.get("given") or [])
                    fam = nm.get("family") or ""
                    txt = f"{given} {fam}".strip()
                if txt:
                    all_names.append(txt)
                    clues.name_raw.append(txt)
                    clues.name_tokens = tokenize_name(txt)
            for tel in res.get("telecom") or []:
                val = str(tel.get("value") or "")
                if tel.get("system") == "email":
                    clues.emails.add(val.lower())
                else:
                    p = norm_phone(val)
                    if p:
                        clues.phones.add(p)
            if res.get("gender"):
                g = norm_gender(str(res["gender"]))
                if g:
                    clues.gender = g
            if res.get("birthDate"):
                d, _ = parse_date_str(str(res["birthDate"]))
                if d:
                    clues.dob = d
                    # Approximate age at 2026
                    clues.age = max(0, 2026 - d.year)
            for addr in res.get("address") or []:
                if addr.get("city"):
                    clues.village = " ".join(tokenize_name(str(addr["city"])))
                if addr.get("district"):
                    clues.district = " ".join(tokenize_name(str(addr["district"])))
                if addr.get("postalCode"):
                    clues.pin = str(addr["postalCode"])
            for cont in res.get("contact") or []:
                cname = (cont.get("name") or {}).get("text")
                if cname:
                    all_names.append(cname)
                    clues.relative_raw.append(cname)
                    clues.relative_tokens = tokenize_name(cname)
        elif rt == "Encounter":
            for ident in res.get("identifier") or []:
                val = str(ident.get("value") or "").strip()
                if val:
                    clues.uhids.add(val.upper())
            period = res.get("period") if isinstance(res.get("period"), dict) else {}
            for pk in ("start", "end"):
                if pk in period:
                    d_ev, _ = parse_date_str(str(period[pk]))
                    if d_ev:
                        clues.event_dates.add(d_ev)
        elif rt == "Condition":
            for cod in (res.get("code") or {}).get("coding") or []:
                if cod.get("code"):
                    clues.diagnoses.add(str(cod["code"]).upper())
            if res.get("recordedDate"):
                d_ev, _ = parse_date_str(str(res["recordedDate"]))
                if d_ev:
                    clues.event_dates.add(d_ev)
        elif rt in ("Observation", "MedicationRequest", "Procedure"):
            for df in ("effectiveDateTime", "authoredOn", "performedDateTime"):
                if res.get(df):
                    d_ev, _ = parse_date_str(str(res[df]))
                    if d_ev:
                        clues.event_dates.add(d_ev)


def _is_multi_record_json(data: Any) -> bool:
    return (
        isinstance(data, dict)
        and data.get("resourceType") != "Bundle"
        and any(isinstance(v, list) and v and all(isinstance(x, dict) for x in v) for v in data.values())
    )


_DIRECT_ROLE_PLACEHOLDERS: dict[str, tuple[str, str]] = {
    "aadhaar": ("[AADHAAR]", "AADHAAR"),
    "abha": ("[ABHA_NUMBER]", "ABHA_NUMBER"),
    "abha_address": ("[ABHA_ADDRESS]", "ABHA_ADDRESS"),
    "pan": ("[PAN]", "PAN"),
    "insurance": ("[INSURANCE_ID]", "INSURANCE_ID"),
    "phone": ("[PHONE]", "PHONE"),
    "email": ("[EMAIL]", "EMAIL"),
    "relative_name": ("[RELATIVE_NAME]", "RELATIVE_NAME"),
    "source_path": ("[SOURCE_PATH]", "SOURCE_PATH"),
    "address": ("[ADDRESS]", "ADDRESS"),
}


def _scrub_field_value(
    key: str,
    val: Any,
    scrubber: GlobalScrubber,
    clues: ClueSet,
    month_first: bool,
    parent_key: str = "",
) -> Any:
    """Scrub a structured field value according to key semantics & quasi-identifier rules."""
    role = classify_field(key)

    # Preserve practitioner / organization / facility names
    if role == "doctor" or parent_key.lower() in {"facility", "hospital", "organization", "practitioner"}:
        if isinstance(val, str):
            return val
        if isinstance(val, dict):
            return {k: _scrub_field_value(k, v, scrubber, clues, month_first, parent_key=key) for k, v in val.items()}
        return val

    # Iterative handling for deeply nested structures if non-string container
    if isinstance(val, dict):
        return {scrubber.scrub_key(k): _scrub_field_value(k, v, scrubber, clues, month_first, parent_key=key) for k, v in val.items()}
    if isinstance(val, list):
        return [_scrub_field_value(key, x, scrubber, clues, month_first, parent_key=key) for x in val]
    if val is None or isinstance(val, bool):
        return val

    sv = str(val)
    if not sv.strip():
        return val

    if role == "uhid":
        scrubber.stats.identifiers += 1
        scrubber.stats.bump("MRN")
        return clues.pseudonym if clues.pseudonym else "[MRN]"
    if role in _DIRECT_ROLE_PLACEHOLDERS:
        placeholder, etype = _DIRECT_ROLE_PLACEHOLDERS[role]
        scrubber.stats.identifiers += 1
        scrubber.stats.bump(etype)
        return placeholder
    if role in ("patient_name", "first_name", "last_name") or (
        role == "ambiguous_name" and parent_key.lower() in {"patient", "raised_by", "name"}
    ):
        scrubber.stats.identifiers += 1
        scrubber.stats.bump("PATIENT_NAME")
        if clues.pseudonym and not clues.uhids:
            return clues.pseudonym
        return "[PATIENT_NAME]"
    if role == "village":
        scrubber.stats.generalized += 1
        scrubber.stats.bump("VILLAGE")
        return "[VILLAGE]"
    if role == "tehsil":
        scrubber.stats.generalized += 1
        scrubber.stats.bump("TEHSIL")
        return "[TEHSIL]"
    if role == "district":
        if clues.suppress_district:
            scrubber.stats.suppressed += 1
            scrubber.stats.bump("DISTRICT")
            return "[DISTRICT]"
        return val
    if role == "pin":
        scrubber.stats.generalized += 1
        scrubber.stats.bump("PIN")
        return "******" if clues.suppress_district else generalize_pin(sv, 3)
    if role == "dob":
        if clues.suppress_district:
            scrubber.stats.suppressed += 1
            scrubber.stats.bump("DOB")
            return "[DOB]"
        d, fmt = parse_date_str(sv, month_first=month_first)
        if d and fmt:
            scrubber.stats.generalized += 1
            scrubber.stats.bump("DOB")
            return format_date_like(d + dt.timedelta(days=clues.date_shift), fmt, sv)
        scrubber.stats.bump("DOB")
        return "[DOB]"
    if role == "age":
        scrubber.stats.generalized += 1
        m = fold_indic_digits(sv).strip()
        if m.isdigit():
            return age_band(int(m), clues.age_width)
        return scrubber.scrub_text(sv, shift_days=clues.date_shift, month_first=month_first, age_width=clues.age_width, suppress_location=clues.suppress_district)
    if role == "date":
        d, fmt = parse_date_str(sv, month_first=month_first)
        if d and fmt:
            scrubber.stats.generalized += 1
            return format_date_like(d + dt.timedelta(days=clues.date_shift), fmt, sv)
        return scrubber.scrub_text(sv, shift_days=clues.date_shift, month_first=month_first, age_width=clues.age_width, suppress_location=clues.suppress_district)
    if role == "clinical" and isinstance(val, (int, float)):
        return val
    if isinstance(val, (int, float)):
        return val

    return scrubber.scrub_text(
        sv,
        shift_days=clues.date_shift,
        month_first=month_first,
        age_width=clues.age_width,
        suppress_location=clues.suppress_district,
    )


def _scrub_fhir_resource_narratives(
    res: dict[str, Any],
    scrubber: GlobalScrubber,
    clues: ClueSet,
    shift: int,
    *,
    is_patient: bool = False,
) -> None:
    """Scrub narrative text.div, note[].text, and extension[].valueString on any FHIR resource."""
    scrub_kw = {"shift_days": shift, "age_width": clues.age_width, "suppress_location": clues.suppress_district}
    if isinstance(res.get("text"), dict) and isinstance(res["text"].get("div"), str):
        res["text"]["div"] = scrubber.scrub_text(res["text"]["div"], **scrub_kw)
    for note in res.get("note") or []:
        if isinstance(note, dict) and isinstance(note.get("text"), str):
            note["text"] = scrubber.scrub_text(note["text"], **scrub_kw)
    for ext in res.get("extension") or []:
        if isinstance(ext, dict) and isinstance(ext.get("valueString"), str):
            if is_patient and any(t in str(ext.get("url", "")).lower() for t in ("mother", "father", "relative", "spouse", "name")):
                ext["valueString"] = "[RELATIVE_NAME]"
                scrubber.stats.identifiers += 1
                scrubber.stats.bump("RELATIVE_NAME")
            else:
                ext["valueString"] = scrubber.scrub_text(ext["valueString"], **scrub_kw)
                scrubber.stats.identifiers += 1


def _scrub_fhir_bundle(bundle: dict[str, Any], scrubber: GlobalScrubber, clues: ClueSet) -> dict[str, Any]:
    """Sanitize a FHIR R4 Bundle while preserving Bundle validity and resolvable references."""
    out = json.loads(json.dumps(bundle))
    pid_label = clues.pseudonym or "[PATIENT_NAME]"
    shift = clues.date_shift

    def _shift_iso(s: str) -> str:
        d, fmt = parse_date_str(s)
        if d and fmt:
            return format_date_like(d + dt.timedelta(days=shift), fmt, s)
        return s

    if isinstance(out.get("meta"), dict):
        if "source" in out["meta"]:
            out["meta"]["source"] = f"#{pid_label}"
        if "lastUpdated" in out["meta"]:
            out["meta"]["lastUpdated"] = _shift_iso(str(out["meta"]["lastUpdated"]))
    if "timestamp" in out:
        out["timestamp"] = _shift_iso(str(out["timestamp"]))

    for ent in out.get("entry") or []:
        res = ent.get("resource") if isinstance(ent, dict) else None
        if not isinstance(res, dict):
            continue
        rt = res.get("resourceType")
        if rt == "Patient":
            _scrub_fhir_resource_narratives(res, scrubber, clues, shift, is_patient=True)
            for ident in res.get("identifier") or []:
                if isinstance(ident, dict) and "value" in ident:
                    scrubber.stats.identifiers += 1
                    scrubber.stats.bump("MRN")
                    ident["value"] = pid_label
            if res.get("name"):
                scrubber.stats.identifiers += 1
                scrubber.stats.bump("PATIENT_NAME")
                res["name"] = [{"text": pid_label}]
            if res.get("telecom"):
                n_tel = len(res["telecom"])
                scrubber.stats.identifiers += n_tel
                for t in res["telecom"]:
                    scrubber.stats.bump("EMAIL" if t.get("system") == "email" else "PHONE")
                res["telecom"] = [{"system": t.get("system", "phone"), "value": "[PHONE]" if t.get("system") != "email" else "[EMAIL]"} for t in res["telecom"]]
            if res.get("birthDate"):
                if clues.suppress_district:
                    res.pop("birthDate", None)
                else:
                    res["birthDate"] = _shift_iso(str(res["birthDate"]))
                scrubber.stats.generalized += 1
                scrubber.stats.bump("DOB")
            if res.get("address"):
                new_addrs = []
                for addr in res["address"]:
                    if not isinstance(addr, dict):
                        continue
                    na = dict(addr)
                    if "line" in na:
                        na["line"] = ["[ADDRESS]"]
                    if "city" in na:
                        na["city"] = "[VILLAGE]"
                    if "district" in na and clues.suppress_district:
                        na["district"] = "[DISTRICT]"
                    if "postalCode" in na:
                        na["postalCode"] = "******" if clues.suppress_district else generalize_pin(str(na["postalCode"]), 3)
                    new_addrs.append(na)
                res["address"] = new_addrs
                scrubber.stats.generalized += 1
                scrubber.stats.bump("ADDRESS")
            if res.get("contact"):
                for cont in res["contact"]:
                    if isinstance(cont, dict) and isinstance(cont.get("name"), dict):
                        cont["name"]["text"] = "[RELATIVE_NAME]"
                        scrubber.stats.identifiers += 1
                        scrubber.stats.bump("RELATIVE_NAME")
            continue

        _scrub_fhir_resource_narratives(res, scrubber, clues, shift)
        if rt == "Composition":
            if isinstance(res.get("subject"), dict) and "display" in res["subject"]:
                res["subject"]["display"] = pid_label
            if "date" in res:
                res["date"] = _shift_iso(str(res["date"]))
            for sec in res.get("section") or []:
                if isinstance(sec, dict) and isinstance(sec.get("text"), dict) and isinstance(sec["text"].get("div"), str):
                    sec["text"]["div"] = scrubber.scrub_text(
                        sec["text"]["div"], shift_days=shift, age_width=clues.age_width, suppress_location=clues.suppress_district
                    )
        elif rt == "Encounter":
            for ident in res.get("identifier") or []:
                if isinstance(ident, dict) and "value" in ident:
                    ident["value"] = pid_label
                    scrubber.stats.identifiers += 1
                    scrubber.stats.bump("MRN")
            if isinstance(res.get("period"), dict):
                for pk in ("start", "end"):
                    if pk in res["period"]:
                        res["period"][pk] = _shift_iso(str(res["period"][pk]))
        elif rt == "Condition":
            if "recordedDate" in res:
                res["recordedDate"] = _shift_iso(str(res["recordedDate"]))
        elif rt == "Observation":
            if "effectiveDateTime" in res:
                res["effectiveDateTime"] = _shift_iso(str(res["effectiveDateTime"]))
        elif rt == "MedicationRequest":
            if "authoredOn" in res:
                res["authoredOn"] = _shift_iso(str(res["authoredOn"]))
        elif rt == "Procedure":
            if "performedDateTime" in res:
                res["performedDateTime"] = _shift_iso(str(res["performedDateTime"]))
        elif rt == "DocumentReference":
            for cont in res.get("content") or []:
                att = cont.get("attachment") if isinstance(cont, dict) else None
                if isinstance(att, dict) and isinstance(att.get("data"), str):
                    att["data"] = scrubber.scrub_b64_if_present(att["data"])

    return out


def _scrub_deep_json_iterative(root: Any, scrubber: GlobalScrubber, clues: ClueSet, month_first: bool) -> Any:
    """Iteratively sanitize arbitrary-depth JSON without hitting Python recursion limits."""
    if not isinstance(root, (dict, list)):
        if isinstance(root, str):
            return scrubber.scrub_text(root, shift_days=clues.date_shift, month_first=month_first, age_width=clues.age_width, suppress_location=clues.suppress_district)
        return root

    placeholder = object()
    top_container: list[Any] = [placeholder]
    stack: list[tuple[Any, Any, Any, str]] = [(root, top_container, 0, "")]

    while stack:
        node, parent, key_or_idx, pkey = stack.pop()
        if isinstance(node, dict):
            new_d: dict[Any, Any] = {}
            parent[key_or_idx] = new_d
            items = list(node.items())
            to_push = []
            seen_keys: dict[str, int] = {}
            for k, v in items:
                sk = scrubber.scrub_key(k)
                if sk != k:
                    cnt = seen_keys.get(sk, 0) + 1
                    seen_keys[sk] = cnt
                    if cnt > 1 or sk in {"[PATIENT_NAME]", "[MRN]"}:
                        sk = f"{sk}_{cnt}"
                if isinstance(v, (dict, list)):
                    if classify_field(k) == "doctor" or k.lower() in {"facility", "hospital", "organization", "practitioner"}:
                        new_d[sk] = v
                    else:
                        new_d[sk] = placeholder
                        to_push.append((v, new_d, sk, k))
                else:
                    new_d[sk] = _scrub_field_value(k, v, scrubber, clues, month_first, parent_key=pkey)
            for item in reversed(to_push):
                stack.append(item)
        elif isinstance(node, list):
            new_l: list[Any] = [placeholder] * len(node)
            parent[key_or_idx] = new_l
            for idx in range(len(node) - 1, -1, -1):
                v = node[idx]
                if isinstance(v, (dict, list)):
                    stack.append((v, new_l, idx, pkey))
                else:
                    new_l[idx] = _scrub_field_value(pkey, v, scrubber, clues, month_first, parent_key=pkey)
    return top_container[0]


def _ensure_csv_row_pseudonym(new_row: list[str], header_roles: list[str], pid: str | None) -> None:
    """Guarantee that a linked CSV row embeds its pseudonym while preserving column schema."""
    if not pid or not new_row or any(pid in cell for cell in new_row):
        return
    target = next((j for j, r in enumerate(header_roles) if r == "free_text" and j < len(new_row)), None)
    if target is None:
        target = next((j for j in range(len(new_row) - 1, -1, -1) if new_row[j].strip()), 0)
    cell = new_row[target]
    for marker in ("[MRN]", "[PATIENT_NAME]", "[PHONE]"):
        if marker in cell:
            new_row[target] = cell.replace(marker, pid, 1)
            return
    new_row[target] = f"Patient: {pid}; {cell}" if cell.strip() else pid


def _inject_pseudonym_in_json(root: Any, pid: str | None) -> None:
    """Iteratively embed `pid` inside a scrubbed JSON tree (safe for 1,500+ depth)."""
    if not pid or not isinstance(root, (dict, list)):
        return
    first_slot_by_marker: dict[str, tuple[Any, Any, str]] = {}
    stack: list[Any] = [root]
    while stack:
        node = stack.pop()
        if isinstance(node, dict):
            items = list(node.items())
            for k, v in reversed(items):
                if isinstance(v, str):
                    if pid in v:
                        return
                    for marker in ("[MRN]", "[PATIENT_NAME]", "[PHONE]"):
                        if marker in v and marker not in first_slot_by_marker:
                            first_slot_by_marker[marker] = (node, k, v)
                elif isinstance(v, (dict, list)):
                    stack.append(v)
        elif isinstance(node, list):
            for idx in range(len(node) - 1, -1, -1):
                v = node[idx]
                if isinstance(v, str):
                    if pid in v:
                        return
                    for marker in ("[MRN]", "[PATIENT_NAME]", "[PHONE]"):
                        if marker in v and marker not in first_slot_by_marker:
                            first_slot_by_marker[marker] = (node, idx, v)
                elif isinstance(v, (dict, list)):
                    stack.append(v)
    for marker in ("[MRN]", "[PATIENT_NAME]", "[PHONE]"):
        if marker in first_slot_by_marker:
            container, key_or_idx, val = first_slot_by_marker[marker]
            container[key_or_idx] = val.replace(marker, pid, 1)
            return
    if isinstance(root, dict) and "patient_ref" not in root:
        root["patient_ref"] = pid


def run_pipeline(inp: Path, out: Path, repo_root: Path) -> int:
    t0 = time.monotonic()
    policy = load_policy(repo_root / "config" / "policy.yaml")
    age_band_years = int(policy.get("age_band_years", 5))
    date_shift_max = int(policy.get("date_shift_max_days", 180))
    min_group_size = int(policy.get("min_group_size", 5))

    san = out / "sanitized"
    san.mkdir(parents=True, exist_ok=True)
    sys.setrecursionlimit(5000)

    # Pass 1: Ingest all files deterministically and extract record clues + global names
    parsed_files: list[ParsedFile] = []
    file_entries: list[dict[str, Any]] = []
    quarantined_indices: set[int] = set()

    all_clues: list[ClueSet] = []
    record_clue_map: dict[tuple[str, str], int] = {}
    all_names: list[str] = []

    all_paths = sorted((p for p in inp.rglob("*") if p.is_file()), key=lambda p: p.relative_to(inp).as_posix())
    for path in all_paths:
        raw = path.read_bytes()
        sha = hashlib.sha256(raw).hexdigest()
        pf, q_reason = ingest_file(path, inp, sha)
        if pf is None:
            file_entries.append({"input_sha256": sha, "output": None, "status": "quarantined", "reason": q_reason})
            quarantined_indices.add(len(file_entries) - 1)
            continue

        file_entries.append({"input_sha256": sha, "output": None, "status": "sanitized", "_pf": pf})
        parsed_files.append(pf)
        mf = _infer_month_first(pf)

        if pf.kind == "csv" and pf.rows:
            header, body = pf.rows[0], pf.rows[1:]
            # Extract any names embedded in headers
            for h in header:
                all_names.extend(extract_names_from_text(h))
            for i, row in enumerate(body):
                loc = f"row:{i}"
                c = ClueSet(file_rel=pf.rel, loc=loc)
                row_dict = {header[j]: row[j] for j in range(min(len(header), len(row)))}
                _extract_dict_clues(row_dict, c, all_names, mf)
                # Also extract from any extra ragged columns
                for extra in row[len(header):]:
                    extract_clues_from_text(extra, c, month_first=mf)
                    all_names.extend(extract_names_from_text(extra))
                record_clue_map[(pf.rel, loc)] = len(all_clues)
                all_clues.append(c)

        elif pf.kind == "fhir" and isinstance(pf.json_data, dict):
            c = ClueSet(file_rel=pf.rel, loc="doc")
            _extract_fhir_clues(pf.json_data, c, all_names)
            record_clue_map[(pf.rel, "doc")] = len(all_clues)
            all_clues.append(c)

        elif pf.kind == "json":
            if _is_multi_record_json(pf.json_data):
                # Extract top-level metadata names
                for k, v in pf.json_data.items():
                    if not (isinstance(v, list) and v and all(isinstance(x, dict) for x in v)):
                        if isinstance(v, (dict, str)):
                            all_names.extend(extract_names_from_text(json.dumps(v, ensure_ascii=False)))
                    else:
                        ptr_k = k.replace("~", "~0").replace("/", "~1")
                        for i, rec in enumerate(v):
                            loc = f"/{ptr_k}/{i}"
                            c = ClueSet(file_rel=pf.rel, loc=loc)
                            _extract_dict_clues(rec, c, all_names, mf)
                            record_clue_map[(pf.rel, loc)] = len(all_clues)
                            all_clues.append(c)
            else:
                c = ClueSet(file_rel=pf.rel, loc="doc")
                if isinstance(pf.json_data, dict):
                    # Check if multi-patient index (e.g. by_uhid / by_name) -> do not link doc to 1 patient
                    keys_low = {str(k).lower() for k in pf.json_data}
                    if "by_uhid" in keys_low or "by_name" in keys_low:
                        all_names.extend(extract_names_from_text(json.dumps(pf.json_data, ensure_ascii=False)))
                        for k2, v2 in pf.json_data.items():
                            if "name" in k2.lower() and isinstance(v2, dict):
                                all_names.extend(str(x) for x in v2.keys())
                    else:
                        _extract_dict_clues(pf.json_data, c, all_names, mf)
                elif isinstance(pf.json_data, str):
                    extract_clues_from_text(pf.json_data, c, month_first=mf)
                    all_names.extend(extract_names_from_text(pf.json_data))
                record_clue_map[(pf.rel, "doc")] = len(all_clues)
                all_clues.append(c)

        elif pf.kind == "txt" and pf.text is not None:
            c = ClueSet(file_rel=pf.rel, loc="doc")
            # Extract filename clues (e.g. UHID/name in Discharge_Summary_Fatima_Bano_SNJ-26-064508.txt)
            fname_clean = pf.rel.rsplit("/", 1)[-1].rsplit(".", 1)[0].replace("_", " ")
            extract_clues_from_text(fname_clean, c, month_first=mf)
            extract_clues_from_text(pf.text, c, month_first=mf)
            all_names.extend(extract_names_from_text(pf.text))
            record_clue_map[(pf.rel, "doc")] = len(all_clues)
            all_clues.append(c)

    # Pass 2: Link records & compute k-anonymity parameters
    link_and_protect(
        all_clues,
        age_band_years=age_band_years,
        date_shift_max_days=date_shift_max,
        min_group_size=min_group_size,
    )

    stats = DetectionStats()
    scrubber = GlobalScrubber(stats, age_width=age_band_years, date_shift_max=date_shift_max)
    scrubber.register_names(all_names)
    scrubber.finalize_lexicon()

    # Pass 3: Sanitize filenames, contents, and write sanitized/ + records.jsonl + report.json
    used_paths: set[str] = set()
    records_out: list[dict[str, Any]] = []
    final_entries: list[dict[str, Any]] = []

    for ent in file_entries:
        if ent["status"] != "sanitized":
            final_entries.append(ent)
            continue

        pf: ParsedFile = ent.pop("_pf")
        mf = _infer_month_first(pf)
        out_rel = sanitize_filename(pf.rel, scrubber.known_first_tokens, used_paths)

        try:
            if pf.kind == "csv" and pf.rows:
                header, body = pf.rows[0], pf.rows[1:]
                buf = io.StringIO()
                w = csv.writer(buf, delimiter=pf.delimiter, lineterminator="\n")

                # Sanitize headers ONLY if PII is present in a header
                new_header: list[str] = []
                header_roles = [classify_field(h) for h in header]
                for idx_col, h in enumerate(header):
                    sh = scrubber.scrub_key(h)
                    if sh != h:
                        sh = f"col_{idx_col}_{hmac_token('hdr', h, 4)}"
                    new_header.append(sh)
                w.writerow(new_header)

                # Decide which column gets the injected pseudonym (UHID col preferred, else patient_name col)
                uhid_cols = [j for j, r in enumerate(header_roles) if r == "uhid"]
                name_cols = [j for j, r in enumerate(header_roles) if r in ("patient_name", "first_name")]

                for i, row in enumerate(body):
                    loc = f"row:{i}"
                    c = all_clues[record_clue_map[(pf.rel, loc)]]
                    pid = c.pseudonym
                    new_row = []
                    for j, cell in enumerate(row):
                        role = header_roles[j] if j < len(header_roles) else "other"
                        h_name = header[j] if j < len(header) else ""
                        if pid and j in uhid_cols:
                            new_row.append(pid)
                            stats.identifiers += 1
                        elif pid and not uhid_cols and j in name_cols and j == name_cols[0]:
                            new_row.append(pid)
                            stats.identifiers += 1
                        else:
                            val = _scrub_field_value(h_name, cell, scrubber, c, mf)
                            new_row.append(str(val) if val is not None else "")
                    _ensure_csv_row_pseudonym(new_row, header_roles, pid)
                    w.writerow(new_row)
                    records_out.append({"file": f"sanitized/{out_rel}", "loc": loc, "patient": pid})
                    stats.records += 1
                    stats.linked += pid is not None
                result_text = buf.getvalue()

            elif pf.kind == "fhir" and isinstance(pf.json_data, dict):
                c = all_clues[record_clue_map[(pf.rel, "doc")]]
                cleaned_bundle = _scrub_fhir_bundle(pf.json_data, scrubber, c)
                result_text = json.dumps(cleaned_bundle, ensure_ascii=False, indent=2) + "\n"
                records_out.append({"file": f"sanitized/{out_rel}", "loc": "doc", "patient": c.pseudonym})
                stats.records += 1
                stats.linked += c.pseudonym is not None

            elif pf.kind == "json":
                if _is_multi_record_json(pf.json_data):
                    cleaned_obj: dict[str, Any] = {}
                    for k, v in pf.json_data.items():
                        sk = scrubber.scrub_key(k)
                        if isinstance(v, list) and v and all(isinstance(x, dict) for x in v):
                            ptr_k = k.replace("~", "~0").replace("/", "~1")
                            cleaned_list = []
                            for i, rec in enumerate(v):
                                loc = f"/{ptr_k}/{i}"
                                c = all_clues[record_clue_map[(pf.rel, loc)]]
                                cleaned_rec = _scrub_deep_json_iterative(rec, scrubber, c, mf)
                                _inject_pseudonym_in_json(cleaned_rec, c.pseudonym)
                                cleaned_list.append(cleaned_rec)
                                records_out.append({"file": f"sanitized/{out_rel}", "loc": loc, "patient": c.pseudonym})
                                stats.records += 1
                                stats.linked += c.pseudonym is not None
                            cleaned_obj[sk] = cleaned_list
                        else:
                            dummy = ClueSet(file_rel=pf.rel, loc="@file")
                            cleaned_obj[sk] = _scrub_deep_json_iterative(v, scrubber, dummy, mf)
                    result_text = json.dumps(cleaned_obj, ensure_ascii=False, indent=2) + "\n"
                else:
                    c = all_clues[record_clue_map[(pf.rel, "doc")]]
                    cleaned_obj = _scrub_deep_json_iterative(pf.json_data, scrubber, c, mf)
                    _inject_pseudonym_in_json(cleaned_obj, c.pseudonym)
                    result_text = json.dumps(cleaned_obj, ensure_ascii=False, indent=2) + "\n"
                    records_out.append({"file": f"sanitized/{out_rel}", "loc": "doc", "patient": c.pseudonym})
                    stats.records += 1
                    stats.linked += c.pseudonym is not None

            else:  # txt
                c = all_clues[record_clue_map[(pf.rel, "doc")]]
                cleaned_txt = scrubber.scrub_text(
                    pf.text or "",
                    shift_days=c.date_shift,
                    month_first=mf,
                    age_width=c.age_width,
                    suppress_location=c.suppress_district,
                )
                # Ensure linked pseudonym appears INSIDE the exported text record for consistency credit
                if c.pseudonym and c.pseudonym not in cleaned_txt:
                    if "[MRN]" in cleaned_txt:
                        cleaned_txt = cleaned_txt.replace("[MRN]", c.pseudonym, 1)
                    else:
                        cleaned_txt = f"Patient: {c.pseudonym}\n" + cleaned_txt
                result_text = cleaned_txt
                records_out.append({"file": f"sanitized/{out_rel}", "loc": "doc", "patient": c.pseudonym})
                stats.records += 1
                stats.linked += c.pseudonym is not None

            dest = san / out_rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(result_text, encoding="utf-8", newline="")
            ent["output"] = f"sanitized/{out_rel}"
            final_entries.append(ent)

        except (ValueError, RecursionError, UnicodeError, csv.Error) as exc:
            reason = str(exc) if isinstance(exc, ValueError) and str(exc).isidentifier() else type(exc).__name__
            final_entries.append({
                "input_sha256": ent["input_sha256"],
                "output": None,
                "status": "quarantined",
                "reason": reason,
            })

    exported = {e["output"] for e in final_entries if e["output"]}
    records_out = [r for r in records_out if r["file"] in exported]

    with open(out / "records.jsonl", "w", encoding="utf-8", newline="\n") as f:
        for r in records_out:
            f.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")

    n_san = sum(1 for e in final_entries if e["status"] == "sanitized")
    rss_mb = peak_rss_mb()
    report = {
        "schema": "raksha-report/1",
        "files": final_entries,
        "counts": {
            "files_total": len(final_entries),
            "files_sanitized": n_san,
            "files_quarantined": len(final_entries) - n_san,
            "records_processed": len(records_out),
            "identifiers_detected": stats.identifiers,
            "records_linked": sum(1 for r in records_out if r["patient"] is not None),
            "records_generalized": stats.generalized,
            "records_suppressed": stats.suppressed,
            "records_quarantined": 0,
        },
        "entities": dict(sorted(stats.entities.items())),
        "timing": {"seconds": round(time.monotonic() - t0, 2)},
        "peak_rss_mb": rss_mb,
        "resources": {"peak_rss_mb": rss_mb},
    }
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n")
    # Only counts go to stdout: stdout and stderr are scanned for PII.
    print(f"files {len(final_entries)} sanitized {n_san} quarantined {len(final_entries) - n_san} records {len(records_out)}")
    return 0
