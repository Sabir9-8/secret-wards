# Secret Wards Privacy Firewall — Team Bug Stupifiers

An offline, deterministic, CPU-only clinical privacy firewall built entirely on the Python 3.10+ standard library. It sanitizes heterogeneous multilingual hospital exports (English, Devanagari Hindi, and Hinglish) into research-ready releases while preserving clinical utility, linking patient records safely across hospitals and scripts, and defending against linkage and re-identification attacks.

---

## 1. Setup & Execution

Matching [`raksha.toml`](raksha.toml):

```bash
# Setup (run once; installs zero external packages)
pip install -r requirements.txt

# Run pipeline (offline, deterministic, keyed by RAKSHA_SECRET)
RAKSHA_SECRET=<your-secret> python main.py --input <input-dir> --output <output-dir>
```

### Verification Commands (with `Secret-Wards-ToolKit` alongside this repo)

```bash
RAKSHA_SECRET=practice python main.py --input ../Secret-Wards-ToolKit/dev/input --output /tmp/sw_out
python3 ../Secret-Wards-ToolKit/validate_output.py --input ../Secret-Wards-ToolKit/dev/input --output /tmp/sw_out
python3 ../Secret-Wards-ToolKit/check_dev.py --output /tmp/sw_out --show 20
python3 ../Secret-Wards-ToolKit/check_submission.py --repo . --input ../Secret-Wards-ToolKit/dev/input --run
```

Privacy parameters are configured in [`config/policy.yaml`](config/policy.yaml): `age_band_years: 5`, `date_shift_max_days: 180`, and `min_group_size: 5`.

---

## 2. Architecture & Pipeline Flow

`main.py` invokes `rakshak.pipeline.run_pipeline`, which executes three deterministic passes over the input directory tree sorted by relative POSIX path:

```text
Input Tree (CSV, TSV, JSON, FHIR R4, TXT, Unsupported/Corrupt)
   │
   ▼  [Pass 1: rakshak/ingest.py + detect.py]
Safe Multi-Encoding Decode & Fail-Safe Quarantine
   │  • Strict UTF-8-BOM, UTF-16, UTF-8, clean cp1252 (rejects corrupt bytes & mislabeled JSON/XML)
   │  • Semantic column/key classification + clue extraction + Pass-1 name lexicon
   ▼  [Pass 2: rakshak/linker.py]
Deterministic 2-Tier Record Linkage & K-Anonymity Risk Sizing
   │  • Tier 1: Direct IDs (UHID, Aadhaar, ABHA, PAN, insurance, email, phone) + conflict guards
   │  • Tier 2: Ambiguity-guarded cross-script Name + secondary clue (DOB/relative/village+age)
   │  • Group-size analysis (min_group_size: 5) for age-band widening & rare-cell suppression
   ▼  [Pass 3: rakshak/detect.py + policy.py + pipeline.py]
Clinical-Trap-Safe Scrubbing & Compliant UTF-8 Export
   │  • Shield doctors, C/O complaints, K/C/O, H/O, doses, labs, eponyms before PII redaction
   │  • Scrub hidden channels: filenames, CSV headers, JSON keys, metadata, base64, FHIR extensions
   │  • Inject HMAC PSN-XXXXXX inside records; shift dates; generalize age bands & 3-digit PINs
   ▼
Output: sanitized/ (UTF-8, \n) + records.jsonl + report.json (counts, SHA-256, timing, RSS)
```

### Module Breakdown (`rakshak/`)

| Module | Responsibility |
|---|---|
| [`policy.py`](rakshak/policy.py) | Parses `config/policy.yaml` without PyYAML; implements keyed HMAC-SHA256 `pseudonym()` (`PSN-XXXXXX` in Crockford base32), `hmac_token()`, non-zero `date_shift_days()`, truthful `age_band()`, and `generalize_pin()`. |
| [`translit.py`](rakshak/translit.py) | NFKC normalization, zero-width stripping, Cyrillic/Greek homoglyph folding, Devanagari digit folding (`०-९` → `0-9`), rule-based Devanagari→Latin transliteration, Hinglish phonetic keys, and pure-Python Jaro-Winkler similarity. |
| [`gazetteer.py`](rakshak/gazetteer.py) | General Indian given/surname sets (Latin and Devanagari), honorifics (`ji`, `sahab`, `didi`, `behenji`, `श्री`, etc.), title prefixes, disease eponym suffixes, and clinical non-PII allowlists. |
| [`ingest.py`](rakshak/ingest.py) | Decodes UTF-8-BOM, UTF-16, UTF-8, and strict `cp1252`; quarantines unsupported extensions, empty files, malformed/deeply recursive errors, corrupt UTF-8/mid-stream BOMs, and mislabeled JSON/XML behind `.csv`/`.tsv`. |
| [`detect.py`](rakshak/detect.py) | Token-level semantic field classifier (`classify_field`), clinical trap shielding (`_shield_clinical`), multilingual direct/indirect PII regexes, `\uXXXX` & base64 payload scrubbing, filename PII detection, and Pass-2 dataset-wide name lexicon sweep. |
| [`linker.py`](rakshak/linker.py) | Homonym-safe 2-tier clustering with strict conflict checks (`are_compatible`) and ambiguity abstention (`patient: null`), plus k-anonymity widening/suppression (`link_and_protect`). |
| [`pipeline.py`](rakshak/pipeline.py) | Orchestrates multi-pass extraction, FHIR R4 Bundle sanitization (preserving resolvable `urn:uuid:` references and clinical resources), iterative arbitrary-depth JSON scrubbing, pseudonym injection inside exported records, and `records.jsonl` / `report.json` generation. |

### How the Six Challenge Traps Are Handled

1. **Language:** Devanagari transliteration + Hinglish phonetic keys + Indic digit folding normalize Latin, Devanagari, and Hinglish variants into shared matching keys.
2. **Meaning:** Doctors (`Dr. <Name>`), symptoms (`C/O <complaint>`), history (`K/C/O`, `H/O`), doses (`650 mg`, `1-0-1`, `BD x 5 days`), lab ranges, hospital wards, and eponyms are shielded prior to scrubbing and restored afterward.
3. **Fragmented Identity:** Tier-2 linkage connects partial records via phonetic name + relative/DOB/village+age only when unambiguous and conflict-free; otherwise emits `"patient": null`.
4. **Re-identification:** Per-patient HMAC date shifts defeat global shift recovery while preserving stay lengths; ages become 5-year (or widened 10/20-year) bands; villages and houses are redacted; PINs become 3-digit prefixes; rare combos (`age >= 80`, rare diagnoses, or cells `< min_group_size`) trigger wider bands and district/DOB/PIN suppression.
5. **Clinical Integrity:** Preserves CSV delimiters/headers, JSON key order/types, ICD/LOINC codes, medication instructions, vitals, and exact date intervals.
6. **Hidden Leakage:** Sanitizes PII inside filenames (recording `input_sha256` → `output` mappings in `report.json`), CSV headers, JSON keys, export metadata, literal `\uXXXX` escapes, base64 attachments, and FHIR `extension` / `Narrative` fields.

---

## 3. Declared Models, Libraries & Offline Compliance

- **LLMs / Generative Models / Pretrained Transformers:** **None.** Zero hosted APIs, zero local LLMs, and zero models >50M parameters.
- **Third-Party Libraries:** **None.** [`requirements.txt`](requirements.txt) uses only the Python 3.10+ standard library, ensuring zero network calls during `run`, deterministic byte-identical output across macOS (`arm64`) and Linux (`aarch64`/`x86_64`), and no wheel/compiler dependencies.
- **Algorithmic Techniques:** Deterministic regular expressions, Unicode normalization, rule-based transliteration, phonetic hashing, Jaro-Winkler string similarity, curated Indian name/clinical word lists, and HMAC-SHA256.

---

## 4. Security Design & Measured Practice Results

- **Secret & Side-Channel Hygiene:** `RAKSHA_SECRET` is read solely from the environment and used only inside HMAC-SHA256 digests (`PSN-XXXXXX` and per-patient shifts). Only aggregate file/record counts are printed to `stdout`; `report.json`, `records.jsonl`, and filenames never emit PII or the secret. No temp files are written to `TMPDIR` or `HOME`, and `--input` is never modified.
- **Fail-Safe Quarantine:** Any file that cannot be decoded cleanly or parsed into its declared format is quarantined and excluded from `sanitized/` and `records.jsonl`.

Measured on `Secret-Wards-ToolKit/dev/input` (200 files: 193 sanitized, 7 quarantined, 822 records):

| Metric | Measured Result |
|---|---|
| Personal Data Leaked (`check_dev.py`) | **0 / 3,230 (0.0%)** across all 16 types, 3 scripts, and 6 channels |
| Planted Canaries Found | **0 / 4** |
| Medical Values Preserved | **4,808 / 4,855 (99.0%)** |
| Patient Linking (`precision / recall / F0.5`) | **1.00 / 0.85 / 0.97** (`0` wrong merges) |
| Raw Copies of Broken/Unsupported Files | **0** |
| Runtime & Peak Memory (`report.json`) | **1.07 s** wall time, **48.6 MB** peak RSS |

---

## 5. Known Limitations

- **Severe spelling drift / partial fragments:** Name spellings diverging beyond phonetic equivalence and the 0.80 Jaro-Winkler threshold, or sparse notes without a corroborating secondary clue, are intentionally left unlinked (`"patient": null`) to avoid false merges.
- **Conservative quarantine:** Files with corrupt byte sequences, truncated JSON, or mislabeled formats (e.g., JSON stored under `.csv`) are quarantined rather than partially recovered.
- **Rule-based free-text generalization:** Unseen clinical phrasing that lacks structural anchors and falls outside known name/clinical word lists may occasionally be over-redacted (~1% medical token loss on practice data) or miss rare unanchored non-standard formats.
