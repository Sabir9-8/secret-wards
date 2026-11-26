#!/usr/bin/env python3
"""Pure-Python L2-regularized Logistic Regression trainer for span disambiguation.

Uses ONLY Python 3.10+ standard library (`math`, `random`, `json`, `resource`, `time`).
Generates ~4,000 completely synthetic labeled span examples (never touching dev/input),
trains a sparse TF-IDF + structural logistic regression model, evaluates on held-out
validation split, and exports `rakshak/models/disambiguator.json` (< 400 KB).
"""
from __future__ import annotations

import json
import math
import random
import resource
import sys
import time
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rakshak.ml_features import extract_features, vectorize_sparse  # noqa: E402

MODEL_OUT = REPO_ROOT / "rakshak" / "models" / "disambiguator.json"


def _peak_rss_mb() -> float:
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return round(r / 1048576 if sys.platform == "darwin" else r / 1024, 2)


def _sigmoid(z: float) -> float:
    if z >= 30.0:
        return 1.0 - 1e-13
    if z <= -30.0:
        return 1e-13
    return 1.0 / (1.0 + math.exp(-z))


def build_synthetic_dataset(seed: int = 42) -> list[tuple[str, str, str, int]]:
    """Build ~4,000 balanced synthetic (span, left_ctx, right_ctx, label) samples."""
    rng = random.Random(seed)
    samples: list[tuple[str, str, str, int]] = []

    # Synthetic Pan-Indian names (independent from dev/input, including rare/unseen stems)
    first_latin = [
        "Aarav", "Bhavna", "Chirag", "Devvrath", "Esha", "Farhan", "Gaurav", "Heena",
        "Ishaan", "Jyotsna", "Kartikeya", "Lavanya", "Madhav", "Nandita", "Omkar", "Pallavi",
        "Qasim", "Raghavendra", "Suhasini", "Tsering", "Uday", "Vasudha", "Wasim", "Yashwant",
        "Zorawar", "Yudhishthir", "Rohan", "Priya", "Sameer", "Ananya", "Vikramaditya", "Meenakshi",
        "Prabhjot", "Venkatesh", "Siddhartha", "Kaveri", "Harshad", "Niharika", "Shubham", "Tanvi",
        "Ramesh", "Suresh", "Mahesh", "Kavita", "Sunita", "Anita", "Rajesh", "Arvind",
        "Mohd. Tariq", "Md. Arif", "Shri Narayan", "Smt. Pushpa", "Kumari Shalini",
    ]
    last_latin = [
        "Kulkarni", "Deshmukh", "Chatterjee", "Sen", "Iyer", "Subramanian", "Nair", "Menon",
        "Reddy", "Naidu", "Gill", "Sandhu", "Bhattacharya", "Mukherjee", "Banerjee", "Patwardhan",
        "Joshi", "Tripathi", "Chaturvedi", "Srivastava", "Agnihotri", "Rastogi", "Saxena", "Awasthi",
        "Qureshi", "Siddiqui", "Ansari", "Khan", "Mirza", "Rathore", "Shekhawat", "Chauhan",
        "Verma", "Sharma", "Gupta", "Pandey", "Mishra", "Tiwari", "Patel", "Rao", "Dorjee", "Wangmo",
    ]
    deva_first = [
        "अनीता", "समीर", "प्रिया", "देवव्रत", "राघवेंद्र", "खगेश्वरी", "युधिष्ठिर", "सुहासिनी",
        "रोहन", "अभिषेक", "नंदिता", "कार्तिकेय", "वंदना", "हरीश", "गौरव", "पल्लवी",
        "फ़रहान", "ज़ोरावर", "मीनाक्षी", "शालिनी", "विवेक", "रमेश", "सुरेश", "कमला",
    ]
    deva_last = [
        "कुलकर्णी", "देशमुख", "त्रिपाठी", "चतुर्वेदी", "नायर", "जोशी", "तिवारी", "शर्मा",
        "वर्मा", "पांडे", "सिद्दीकी", "अंसारी", "कुरैशी", "पटेल", "देवी", "सिंह", "चौहान", "राव",
    ]

    honorifics = ["ji", "sahab", "saheb", "didi", "behenji", "bhaiya", "bhai", "amma", "chacha", "babu"]
    deva_honorifics = ["जी", "साहब", "दीदी", "बहनजी", "भैया", "अम्मा"]

    loc_prefixes = ["Mohalla", "Gram", "Vill.", "Sector", "Ward", "Colony", "Near", "Opp.", "Behind"]
    loc_roots = [
        "Khandala", "Lonavala", "Qazipura", "Rampur", "Biswan", "Sidhauli", "Nawabganj", "Malihabad",
        "Sandila", "Mahmudabad", "Safipur", "Bachhrawan", "Canarypur", "Govindnagar", "Tilak Nagar",
        "Civil Lines", "Station Road", "Old Post Office", "Jama Masjid", "Bus Stand", "Rajendra Nagar",
        "Subhash Nagar", "Indira Colony", "Gomti Nagar", "Shivpuri", "Ashok Vihar", "Kothrud", "Baner",
    ]
    loc_suffixes = ["", " Kalan", " Khurd", " Bazar", " Chowk", " Marg", " Bagh"]
    deva_locs = [
        "ग्राम बछरावां", "मोहल्ला काजीपुरा", "ग्राम रामपुर कलां", "वार्ड 9", "कैनरीपुर",
        "तिलक नगर", "मकान नं. 42", "सिविल लाइंस", "हनुमान मंदिर के पास", "नवाबगंज",
    ]

    # Synthetic canary-like random syllables so model relies on shape/context, not memorized names
    syllables = ["ka", "ra", "vi", "na", "su", "ma", "ta", "de", "sha", "pa", "va", "ya", "la", "ga", "dha", "ni"]

    def rand_synth_name() -> str:
        w1 = "".join(rng.choices(syllables, k=rng.randint(2, 4))).capitalize()
        w2 = "".join(rng.choices(syllables, k=rng.randint(2, 4))).capitalize()
        return f"{w1} {w2}"

    # ---------------- POSITIVE SAMPLES (PII_ENTITY = 1) ----------------
    # 1. Unanchored & labeled Latin full names (850)
    pii_lefts = [
        "", "Patient: ", "Patient Name: ", "Name: ", "Pt. ", "Beneficiary: ",
        "Report of ", "Called by ", "Witnessed by ", "Referral: ",
        "18:40 - ", "OPD: ", "Ward handover: ", "Emergency entry: ",
    ]
    pii_rights = [
        "", " (46/M)", " (attendant)", " S/o Harish", " W/o Vinod",
        ", 38 Y/F", " admitted with fever", " ka BP 138/88", " reported today",
        " brought to casualty", " (Mob 9876543210)",
    ]
    for i in range(850):
        if i % 5 == 0:
            name = rand_synth_name()
        else:
            name = f"{rng.choice(first_latin)} {rng.choice(last_latin)}"
        lc = rng.choice(pii_lefts)
        rc = rng.choice(pii_rights)
        samples.append((name, lc, rc, 1))

    # 2. Relatives & C/O <Person> disambiguation (350)
    rel_lefts = ["S/o ", "D/o ", "W/o ", "C/O ", "Attendant: ", "Guardian: ", "Caregiver: ", "Father: ", "Husband: "]
    rel_rights = ["", " sitting bedside", " (attendant)", ", H.No. 14", ", Ward 9", " informed", ", Mob 9811177001"]
    for i in range(350):
        name = rand_synth_name() if i % 4 == 0 else f"{rng.choice(first_latin)} {rng.choice(last_latin)}"
        samples.append((name, rng.choice(rel_lefts), rng.choice(rel_rights), 1))

    # 3. Devanagari names (350)
    deva_lefts = ["", "मरीज़ का नाम: ", "नाम: ", "रोगी का नाम: ", "पिता का नाम: ", "पति का नाम: ", "S/o ", "W/o "]
    deva_rights = ["", " को बुखार है", " की रिपोर्ट", " जी का रक्तचाप", " (46 वर्ष)", " से मिलें"]
    for _ in range(350):
        name = f"{rng.choice(deva_first)} {rng.choice(deva_last)}"
        samples.append((name, rng.choice(deva_lefts), rng.choice(deva_rights), 1))

    # 4. Hinglish honorific & single-name person mentions (200)
    for i in range(200):
        fn = rng.choice(first_latin).split()[-1]
        if i % 3 == 0:
            # Single-token anchored patient/relative name
            samples.append((fn, rng.choice(["Patient: ", "Name: ", "S/o ", "W/o ", "Report of "]), rng.choice(["", ", W/o Kamal", " (42/M)"]), 1))
        elif rng.random() < 0.5:
            span = f"{fn} {rng.choice(honorifics)}"
            samples.append((span, rng.choice(["", "Note: "]), rng.choice([" ka BP 120/80", " ki tabiyat", " ko fever"]), 1))
        else:
            span = f"{rng.choice(deva_first)} {rng.choice(deva_honorifics)}"
            samples.append((span, "", rng.choice([" को बुखार", " का बीपी"]), 1))

    # 5. Localities, villages, houses, landmarks (including single-word towns + PIN contexts) (300)
    single_towns = [
        "Khandala", "Lonavala", "Qazipura", "Rampur", "Biswan", "Sidhauli", "Nawabganj",
        "Malihabad", "Sandila", "Mahmudabad", "Safipur", "Bachhrawan", "Canarypur",
        "Kothrud", "Baner", "Aliganj", "Gomtinagar", "Hazratganj", "Pimpri", "Chinchwad",
    ]
    for i in range(300):
        rand_pin = f"{rng.randint(110, 859)}{rng.randint(100, 999)}"
        if i % 5 == 0:
            span = rng.choice(single_towns)
            samples.append((span, rng.choice(["Address: ", "Vill. ", "Location: ", "City: "]), rng.choice([f", {rand_pin}", f" - {rand_pin}", ", Pune", ", Sitapur"]), 1))
        elif i % 4 == 0:
            span = rng.choice(deva_locs)
            samples.append((span, rng.choice(["पता: ", "ग्राम: ", ""]), "", 1))
        elif i % 3 == 0:
            span = f"H.No. {rng.randint(1, 500)}, {rng.choice(loc_prefixes)} {rng.choice(loc_roots)}"
            samples.append((span, rng.choice(["Address: ", ""]), "", 1))
        else:
            span = f"{rng.choice(loc_prefixes)} {rng.choice(loc_roots)}{rng.choice(loc_suffixes)}".strip()
            samples.append((span, rng.choice(["", "Address: ", "Vill. ", ", "]), rng.choice(["", ", Sitapur", f", {rand_pin}"]), 1))

    # ---------------- NEGATIVE SAMPLES (CLINICAL_OR_SAFE = 0) ----------------
    # 1. Doctors (both "Dr. <Name>" and "<Name>" with left_ctx="Dr. ") (550)
    doc_lefts = ["Dr. ", "Dr ", "Consultant Dr. ", "Reviewed by Dr. ", "Advised by Dr. ", "डॉ. "]
    doc_rights = ["", " (MD Medicine)", " (MS Surgery)", " reviewed", " advised Dolo 650", " on round", " is nodal officer"]
    for i in range(550):
        if i % 6 == 0:
            dname = f"{rng.choice(deva_first)} {rng.choice(deva_last)}"
            if i % 12 == 0:
                samples.append((f"डॉ. {dname}", "", rng.choice(doc_rights), 0))
            else:
                samples.append((dname, "डॉ. ", rng.choice(doc_rights), 0))
        else:
            raw_name = rand_synth_name() if i % 5 == 0 else f"{rng.choice(first_latin)} {rng.choice(last_latin)}"
            if i % 2 == 0:
                samples.append((raw_name, rng.choice(doc_lefts[:-1]), rng.choice(doc_rights), 0))
            else:
                samples.append((f"Dr. {raw_name}", rng.choice(["", "Consult: ", "By "]), rng.choice(doc_rights), 0))

    # 2. Disease eponyms & eponymous clinical items (450)
    eponym_bases = [
        ("Parkinson's", "disease"), ("Alzheimer's", "disease"), ("Addison's", "disease"),
        ("Crohn's", "disease"), ("Graves'", "disease"), ("Bell's", "palsy"),
        ("Cushing's", "syndrome"), ("Hodgkin", "lymphoma"), ("Hashimoto's", "thyroiditis"),
        ("Raynaud's", "phenomenon"), ("Ringer's", "lactate"), ("Foley", "catheter"),
        ("Koch's", "abdomen"), ("Babinski", "sign"), ("Murphy's", "sign"),
        ("Wilson's", "disease"), ("Hansen's", "disease"), ("Creutzfeldt-Jakob", "disease"),
        ("Wilms", "tumor"), ("Menetrier's", "disease"), ("Paget's", "disease"),
    ]
    for i in range(450):
        first_part, second_part = rng.choice(eponym_bases)
        full = f"{first_part} {second_part}"
        if i % 3 == 0:
            # Bare first part with eponym noun in right_ctx
            samples.append((first_part, rng.choice(["K/C/O ", "H/O ", ""]), f" {second_part}", 0))
        else:
            samples.append((full, rng.choice(["", "K/C/O ", "H/O ", "Diagnosis: ", "Differential: "]), rng.choice(["", " noted", " ruled out", " workup"]), 0))

    # 3. Complaints, histories, diagnoses (C/O <symptom>, K/C/O, H/O) (400)
    complaints = [
        "fever", "fever and chills", "chest pain", "pain abdomen", "breathlessness",
        "acute gastritis", "cough", "generalised weakness", "headache", "vomiting",
        "high grade fever", "dry cough", "dyspnea", "body ache", "pedal edema",
    ]
    histories = [
        "T2DM", "type 2 diabetes", "type 2 diabetes mellitus", "HTN", "essential hypertension",
        "bronchial asthma", "hypothyroidism", "PTB", "rheumatic heart disease", "chronic kidney disease",
        "coronary artery disease", "acute viral hepatitis", "enteric fever", "dengue hemorrhagic fever",
    ]
    deva_clinical = [
        "बुखार", "पेशाब में संक्रमण", "फेफड़ों की टीबी", "निमोनिया", "मलेरिया",
        "मधुमेह", "उच्च रक्तचाप", "दमा", "सिरदर्द", "खांसी",
    ]
    for i in range(400):
        if i % 5 == 0:
            samples.append((rng.choice(deva_clinical), rng.choice(["निदान: ", "मरीज़ को ", ""]), "", 0))
        elif i % 2 == 0:
            samples.append((rng.choice(complaints), rng.choice(["C/O ", "c/o ", "Complaints: "]), rng.choice(["", " x 5 days", " since 3d"]), 0))
        else:
            samples.append((rng.choice(histories), rng.choice(["K/C/O ", "H/O ", "Diagnosis: "]), rng.choice(["", " x 8 yrs", " on Rx"]), 0))

    # 4. Medications, doses, frequencies, vitals, labs, ICD codes (400)
    drugs = [
        "Dolo 650", "Paracetamol", "Metformin", "Ramipril", "Amlodipine", "Pantoprazole",
        "Ceftriaxone", "Azithromycin", "Doxycycline", "Levothyroxine", "Salbutamol inhaler",
        "Telmisartan", "Glimepiride", "Atorvastatin", "Aspirin", "Clopidogrel", "ORS",
    ]
    doses_freqs = [
        "1-0-1", "0-1-0", "1-1-1", "0-0-1", "500 mg", "650 mg", "40 mg", "5 mg",
        "BD x 5 days", "TDS x 7d", "OD x 30 days", "BP 134/86", "BP 126/82", "SpO2 97%",
        "Hb 13.4 g/dL", "HbA1c 7.4%", "WBC 11,400/uL", "Creatinine 1.1 mg/dL", "ICD-10: E11.9",
        "101.2°F", "70-140", "150000-410000", "4000-11000",
    ]
    for i in range(400):
        if i % 2 == 0:
            d = rng.choice(drugs)
            samples.append((d, rng.choice(["Tab ", "Cap ", "Inj ", "Rx: ", ""]), rng.choice([" 500 mg", " 1-0-1", " BD x 5 days", ""]), 0))
        else:
            samples.append((rng.choice(doses_freqs), rng.choice(["", "Vitals: ", "Lab: "]), "", 0))

    # 5. Hospital wards, beds, roles, protocols (200)
    wards_safe = [
        "Ward 4 (Medicine)", "Ward 8 (Surgery)", "Ward 5 (Neurology)", "Ward 2", "Bed 12",
        "ICU", "NICU", "ASHA workers", "OPD", "IPD", "Pathology Lab", "District Hospital",
        "Charitable Hospital", "Standard Treatment Protocol", "eHospital OPD module",
        "Low salt diet", "Review after 2 weeks", "NS1 antigen", "Platelet Count", "Serum Sodium",
    ]
    for _ in range(200):
        samples.append((rng.choice(wards_safe), rng.choice(["", "Admitted to ", "Dept: "]), "", 0))

    rng.shuffle(samples)
    return samples


def train_and_export() -> int:
    t0 = time.monotonic()
    dataset = build_synthetic_dataset(seed=42)
    split_idx = int(0.8 * len(dataset))
    train_data = dataset[:split_idx]
    val_data = dataset[split_idx:]

    # 1. Extract raw features & build capped vocabulary (train split only)
    train_raw: list[tuple[dict[str, float], dict[str, float], int]] = []
    pos_df: Counter[str] = Counter()
    neg_df: Counter[str] = Counter()
    all_struct: set[str] = set()

    for span, lc, rc, label in train_data:
        ng, st = extract_features(span, lc, rc)
        train_raw.append((ng, st, label))
        all_struct.update(st.keys())
        counter = pos_df if label == 1 else neg_df
        for k in ng:
            counter[k] += 1

    n_pos = sum(1 for _, _, y in train_raw if y == 1)
    n_neg = len(train_raw) - n_pos

    # Score n-gram discriminativeness cleanly (log-odds / mutual-info proxy) with min_df=2
    ng_scores: list[tuple[float, str]] = []
    all_ng = sorted(set(pos_df) | set(neg_df))
    for k in all_ng:
        df = pos_df[k] + neg_df[k]
        if df < 2:
            continue
        p1 = (pos_df[k] + 0.5) / (n_pos + 1.0)
        p0 = (neg_df[k] + 0.5) / (n_neg + 1.0)
        score = abs(math.log(p1 / p0)) * math.log1p(df)
        ng_scores.append((score, k))
    ng_scores.sort(key=lambda x: (-x[0], x[1]))

    max_features = 3500
    struct_sorted = sorted(all_struct)
    ng_budget = max(0, max_features - len(struct_sorted))
    chosen_ng = [k for _, k in ng_scores[:ng_budget]]
    vocab_keys = struct_sorted + sorted(chosen_ng)
    vocab = {k: i for i, k in enumerate(vocab_keys)}

    # 2. Compute IDF weights
    n_train = len(train_raw)
    idf: list[float] = []
    for k in vocab_keys:
        if k.startswith("st:"):
            idf.append(1.0)
        else:
            df = pos_df[k] + neg_df[k]
            idf.append(round(math.log((1.0 + n_train) / (1.0 + df)) + 1.0, 4))

    # Vectorize train and validation splits
    train_vecs = [(vectorize_sparse(ng, st, vocab, idf), y) for ng, st, y in train_raw]
    val_vecs = []
    for span, lc, rc, y in val_data:
        ng, st = extract_features(span, lc, rc)
        val_vecs.append((vectorize_sparse(ng, st, vocab, idf), y))

    # 3. Deterministic L2-regularized Logistic Regression SGD (20 epochs, learning rate decay)
    weights = [0.0] * len(vocab)
    bias = 0.0
    rng = random.Random(42)
    epochs = 20
    lr0 = 0.40
    l2 = 1e-4

    indices = list(range(len(train_vecs)))
    for epoch in range(epochs):
        lr = lr0 / (1.0 + 0.15 * epoch)
        rng.shuffle(indices)
        for idx_sample in indices:
            vec, y = train_vecs[idx_sample]
            z = bias
            for j, val in vec.items():
                z += weights[j] * val
            pred = _sigmoid(z)
            err = pred - y
            bias -= lr * err
            for j, val in vec.items():
                weights[j] -= lr * (err * val + l2 * weights[j])

    # Round weights for compact deterministic JSON artifact
    weights_rounded = [round(w, 4) for w in weights]
    bias_rounded = round(bias, 4)
    threshold = 0.72

    # 4. Evaluate on validation set using exported rounded weights
    tp = fp = fn = tn = 0
    for vec, y in val_vecs:
        z = bias_rounded
        for j, val in vec.items():
            z += weights_rounded[j] * val
        prob = _sigmoid(z)
        pred_label = 1 if prob >= threshold else 0
        if pred_label == 1 and y == 1:
            tp += 1
        elif pred_label == 1 and y == 0:
            fp += 1
        elif pred_label == 0 and y == 1:
            fn += 1
        else:
            tn += 1

    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2.0 * prec * rec / (prec + rec)) if (prec + rec) else 0.0

    # 5. Export compact JSON artifact
    MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
    artifact = {
        "version": "1.0",
        "threshold": threshold,
        "bias": bias_rounded,
        "vocab": vocab,
        "idf": idf,
        "weights": weights_rounded,
    }
    MODEL_OUT.write_text(json.dumps(artifact, ensure_ascii=False, separators=(",", ":")), encoding="utf-8", newline="\n")
    elapsed = time.monotonic() - t0
    size_kb = MODEL_OUT.stat().st_size / 1024.0

    ranked = sorted(zip(weights_rounded, vocab_keys), key=lambda x: (-x[0], x[1]))
    print(f"Dataset: total={len(dataset)} (train={len(train_data)}, val={len(val_data)}) | vocab={len(vocab)}")
    print(f"Validation @ threshold={threshold}: Precision={prec:.4f}, Recall={rec:.4f}, F1={f1:.4f} (TP={tp}, FP={fp}, FN={fn}, TN={tn})")
    print(f"Saved {MODEL_OUT.relative_to(REPO_ROOT)} ({size_kb:.1f} KB) in {elapsed:.2f}s | peak_rss={_peak_rss_mb():.2f} MB")
    print("Top 15 positive (PII) features:", [(k, w) for w, k in ranked[:15]])
    print("Top 15 negative (CLINICAL/SAFE) features:", [(k, w) for w, k in ranked[-15:][::-1]])
    return 0


if __name__ == "__main__":
    raise SystemExit(train_and_export())
