# Secret Wards — Antigravity Workspace Rules

## 1. Non-Negotiable Ground Rules
- No LLMs or heavy models: Never import or use hosted/local LLMs, generative models, or pretrained transformers above 50M parameters in runtime code or requirements.txt. Only deterministic rules, gazetteers, classical ML (e.g., CRFs, spaCy rule matchers), or small self-trained models are allowed; declare any model in README.md.
- Offline & CPU-only: Runtime (run in raksha.toml) must work with zero network calls on 2 CPUs and 4 GB RAM. Do not use requests, urllib, httpx, or online downloads during run.
- No hardcoding or memorizing dev data: Never hardcode practice names, file names, column headers, or dev/truth/truth.json logic. Generalize for unseen/chaos datasets.
- Zero leaks & fail-safe quarantine: Never write RAKSHA_SECRET or PII to outputs, report.json, records.jsonl, logs, stdout/stderr, temp files, or error messages. Never copy broken/unsupported/uncleanable files raw into sanitized/—quarantine them. Never modify --input.
- Determinism & secret keying: Sort files, keys, and clusters deterministically (never rely on unordered set iteration or random without seeding). Derive pseudonyms and per-patient date shifts from RAKSHA_SECRET via HMAC.

## 2. Privacy & Clinical Utility Rules
- Multilingual & hidden PII: Detect PII across English, Devanagari Hindi, and Hinglish (names, honorifics, relatives S/o/D/o/W/o/C/O, Indian IDs, phones, emails, addresses) as well as filenames, CSV headers, JSON keys, metadata, and encoded/base64 payloads.
- Meaning traps (preserve clinical data): Keep doctors (Dr. <name>), complaints/history (C/O fever, K/C/O, H/O), dosages/frequencies (650mg, 1-0-1, BD x 5 days), vitals, labs, and disease eponyms (e.g., Parkinson's disease).
- Record linkage: Link only when evidence is strong; output "patient": null when uncertain. Never merge homonyms (different people sharing a name). Embed the assigned pseudonym inside the cleaned record as well as records.jsonl.
- Quasi-identifiers: Apply consistent per-patient date shifts (preserving stay lengths and intervals), truthful age bands, location generalization, and rare-value suppression without destroying clinical utility. Keep CSV/JSON/FHIR R4 valid and UTF-8.

## 3. Verification Commands (Run After Code Edits)
Assume the reference kit is at ../Secret-Wards-ToolKit and submission repo is at . (with .venv active):
- Run pipeline on practice data:
  RAKSHA_SECRET=practice python main.py --input ../Secret-Wards-ToolKit/dev/input --output /tmp/sw_out
- Check output contract:
  python3 ../Secret-Wards-ToolKit/validate_output.py --input ../Secret-Wards-ToolKit/dev/input --output /tmp/sw_out
- Check leaks, preservation, and linking:
  python3 ../Secret-Wards-ToolKit/check_dev.py --output /tmp/sw_out --show 20
- Check repo & 3-run determinism/secret behavior:
  python3 ../Secret-Wards-ToolKit/check_submission.py --repo . --input ../Secret-Wards-ToolKit/dev/input --run
