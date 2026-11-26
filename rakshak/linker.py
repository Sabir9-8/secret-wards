"""Deterministic evidence-based record linkage (homonym-safe) and k-anonymity risk mitigation."""
from __future__ import annotations

import datetime as dt
from collections import Counter
from dataclasses import dataclass, field

from .detect import ClueSet
from .policy import age_band, date_shift_days, pseudonym
from .translit import names_match


@dataclass
class Cluster:
    cid: int
    members: list[int] = field(default_factory=list)
    uhids: set[str] = field(default_factory=set)
    aadhaars: set[str] = field(default_factory=set)
    abhas: set[str] = field(default_factory=set)
    pans: set[str] = field(default_factory=set)
    insurances: set[str] = field(default_factory=set)
    phones: set[str] = field(default_factory=set)
    emails: set[str] = field(default_factory=set)
    names: list[list[str]] = field(default_factory=list)
    relatives: list[list[str]] = field(default_factory=list)
    dobs: set[dt.date] = field(default_factory=set)
    ages: list[int] = field(default_factory=list)
    genders: set[str] = field(default_factory=set)
    villages: set[str] = field(default_factory=set)
    districts: set[str] = field(default_factory=set)
    diagnoses: set[str] = field(default_factory=set)

    def add_clues(self, idx: int, c: ClueSet) -> None:
        self.members.append(idx)
        self.uhids |= c.uhids
        self.aadhaars |= c.aadhaars
        self.abhas |= c.abhas
        self.pans |= c.pans
        self.insurances |= c.insurances
        self.phones |= c.phones
        self.emails |= c.emails
        if c.name_tokens and c.name_tokens not in self.names:
            self.names.append(c.name_tokens)
        if c.relative_tokens and c.relative_tokens not in self.relatives:
            self.relatives.append(c.relative_tokens)
        if c.dob:
            self.dobs.add(c.dob)
        if c.age is not None:
            self.ages.append(c.age)
        if c.gender:
            self.genders.add(c.gender)
        if c.village:
            self.villages.add(c.village)
        if c.district:
            self.districts.add(c.district)
        self.diagnoses |= c.diagnoses

    def merge_from(self, other: Cluster) -> None:
        self.members.extend(other.members)
        self.uhids |= other.uhids
        self.aadhaars |= other.aadhaars
        self.abhas |= other.abhas
        self.pans |= other.pans
        self.insurances |= other.insurances
        self.phones |= other.phones
        self.emails |= other.emails
        for n in other.names:
            if n not in self.names:
                self.names.append(n)
        for r in other.relatives:
            if r not in self.relatives:
                self.relatives.append(r)
        self.dobs |= other.dobs
        self.ages.extend(other.ages)
        self.genders |= other.genders
        self.villages |= other.villages
        self.districts |= other.districts
        self.diagnoses |= other.diagnoses


def _best_name_sim(names_a: list[list[str]], names_b: list[list[str]]) -> float:
    best = 0.0
    for na in names_a:
        for nb in names_b:
            s = names_match(na, nb)
            if s > best:
                best = s
    return best


def are_compatible(a: Cluster, b: Cluster) -> bool:
    """Strict conflict guard to prevent homonym merges and transitive over-merging."""
    if a.genders and b.genders and a.genders.isdisjoint(b.genders):
        return False
    if a.dobs and b.dobs and a.dobs.isdisjoint(b.dobs):
        return False
    if a.ages and b.ages:
        if min(abs(x - y) for x in a.ages for y in b.ages) > 4:
            return False
    if a.aadhaars and b.aadhaars and a.aadhaars.isdisjoint(b.aadhaars):
        return False
    if a.pans and b.pans and a.pans.isdisjoint(b.pans):
        return False
    if a.abhas and b.abhas and a.abhas.isdisjoint(b.abhas):
        return False
    if a.districts and b.districts and a.districts.isdisjoint(b.districts):
        return False
    if a.names and b.names and _best_name_sim(a.names, b.names) < 0.72:
        return False
    if a.relatives and b.relatives and _best_name_sim(a.relatives, b.relatives) < 0.72:
        return False
    return True


def _has_strong_evidence(c: ClueSet) -> bool:
    return bool(
        c.uhids or c.aadhaars or c.abhas or c.pans or c.insurances or c.emails or c.phones
        or (len(c.name_tokens) >= 2 and (c.dob or c.relative_tokens or (c.village and c.age is not None)))
    )


def link_and_protect(
    clues_list: list[ClueSet],
    age_band_years: int = 5,
    date_shift_max_days: int = 180,
    min_group_size: int = 5,
) -> None:
    """Cluster patient records deterministically, assign HMAC pseudonyms, and apply k-anonymity."""
    # Only cluster records that carry at least one patient-identifying clue
    active_indices = [
        i for i, c in enumerate(clues_list)
        if (c.uhids or c.aadhaars or c.abhas or c.pans or c.insurances
            or c.phones or c.emails or c.name_tokens)
    ]

    clusters: dict[int, Cluster] = {}
    rec_to_cid: dict[int, int] = {}
    for idx in active_indices:
        cl = Cluster(cid=idx)
        cl.add_clues(idx, clues_list[idx])
        clusters[idx] = cl
        rec_to_cid[idx] = idx

    def _merge_cids(cid_a: int, cid_b: int) -> int:
        lo, hi = (cid_a, cid_b) if cid_a < cid_b else (cid_b, cid_a)
        if lo == hi or lo not in clusters or hi not in clusters:
            return lo
        ca, cb = clusters[lo], clusters[hi]
        ca.merge_from(cb)
        for m in cb.members:
            rec_to_cid[m] = lo
        del clusters[hi]
        return lo

    # Tier 1: Merge on unique direct identifiers (UHID, Aadhaar, ABHA, PAN, insurance, email, phone)
    changed = True
    while changed:
        changed = False
        id_buckets: dict[tuple[str, str], list[int]] = {}
        for cid in sorted(clusters):
            cl = clusters[cid]
            for u in sorted(cl.uhids):
                id_buckets.setdefault(("uhid", u), []).append(cid)
            for a in sorted(cl.aadhaars):
                id_buckets.setdefault(("aadhaar", a), []).append(cid)
            for ab in sorted(cl.abhas):
                id_buckets.setdefault(("abha", ab), []).append(cid)
            for p in sorted(cl.pans):
                id_buckets.setdefault(("pan", p), []).append(cid)
            for ins in sorted(cl.insurances):
                id_buckets.setdefault(("ins", ins), []).append(cid)
            for em in sorted(cl.emails):
                id_buckets.setdefault(("email", em), []).append(cid)
            for ph in sorted(cl.phones):
                id_buckets.setdefault(("phone", ph), []).append(cid)

        for key in sorted(id_buckets):
            cids = [c for c in sorted(set(id_buckets[key])) if c in clusters]
            if len(cids) < 2:
                continue
            kind = key[0]
            for i in range(len(cids)):
                cid_a = cids[i]
                if cid_a not in clusters:
                    continue
                for cid_b in cids[i + 1:]:
                    if cid_a not in clusters or cid_b not in clusters or cid_a == cid_b:
                        continue
                    ca, cb = clusters[cid_a], clusters[cid_b]
                    if not are_compatible(ca, cb):
                        continue
                    # For phones, require name compatibility if both have names
                    if kind == "phone" and ca.names and cb.names and _best_name_sim(ca.names, cb.names) < 0.78:
                        continue
                    cid_a = _merge_cids(cid_a, cid_b)
                    changed = True

    # Tier 2: Cross-hospital / fragmented-note linkage via Name + Secondary Clues (ambiguity-guarded)
    def _pair_score(ca: Cluster, cb: Cluster) -> float:
        if not are_compatible(ca, cb):
            return 0.0
        nsim = _best_name_sim(ca.names, cb.names)
        if nsim < 0.78:
            return 0.0
        sec = 0.0
        if ca.dobs and cb.dobs and not ca.dobs.isdisjoint(cb.dobs):
            sec += 3.0
        if ca.relatives and cb.relatives and _best_name_sim(ca.relatives, cb.relatives) >= 0.78:
            sec += 2.5
        if ca.villages and cb.villages and not ca.villages.isdisjoint(cb.villages):
            if ca.ages and cb.ages and min(abs(x - y) for x in ca.ages for y in cb.ages) <= 2:
                sec += 2.2
            elif not ca.ages or not cb.ages:
                sec += 1.8
            else:
                sec += 1.2
        if ca.districts and cb.districts and not ca.districts.isdisjoint(cb.districts):
            if nsim >= 0.90 and any(len(n) >= 2 for n in ca.names) and any(len(n) >= 2 for n in cb.names):
                if ca.ages and cb.ages and min(abs(x - y) for x in ca.ages for y in cb.ages) <= 2:
                    sec += 2.2
                elif ca.genders and cb.genders and not ca.genders.isdisjoint(cb.genders) and (not ca.ages or not cb.ages):
                    sec += 1.8
        if ca.phones and cb.phones and not ca.phones.isdisjoint(cb.phones):
            sec += 3.0
        if sec < 2.0:
            return 0.0
        return nsim + sec

    changed = True
    while changed:
        changed = False
        cids_sorted = sorted(clusters)
        for cid_a in cids_sorted:
            if cid_a not in clusters:
                continue
            ca = clusters[cid_a]
            if not ca.names:
                continue
            # Find all compatible candidate partners with strong Name + Secondary clue
            cands: list[tuple[float, int]] = []
            for cid_b in cids_sorted:
                if cid_b == cid_a or cid_b not in clusters:
                    continue
                sc = _pair_score(ca, clusters[cid_b])
                if sc > 0.0:
                    cands.append((sc, cid_b))
            if not cands:
                continue
            cands.sort(key=lambda x: (-x[0], x[1]))
            best_sc, best_b = cands[0]
            # Ambiguity guard: if a second distinct incompatible cluster has nearly the same score, do NOT guess
            if len(cands) > 1:
                second_sc, second_b = cands[1]
                if second_sc >= best_sc - 0.5 and not are_compatible(clusters[best_b], clusters[second_b]):
                    continue
            _merge_cids(cid_a, best_b)
            changed = True

    # Assign canonical deterministic patient_key, pseudonym, date_shift, and k-anonymity params
    # First identify unambiguous linkable clusters
    diag_counts: Counter[str] = Counter()
    cell_counts: Counter[tuple[str, str]] = Counter()
    cluster_meta: dict[int, tuple[str | None, str | None]] = {}

    for cid, cl in sorted(clusters.items()):
        # A cluster is linkable if size >= 2 OR its single record has strong evidence
        is_linkable = (len(cl.members) >= 2) or (len(cl.members) == 1 and _has_strong_evidence(clues_list[cl.members[0]]))
        if not is_linkable:
            continue
        for d in cl.diagnoses:
            diag_counts[d] += 1
        rep_age = sorted(cl.ages)[len(cl.ages) // 2] if cl.ages else None
        rep_dist = sorted(cl.districts)[0] if cl.districts else None
        ab = age_band(rep_age, age_band_years) if rep_age is not None else None
        cluster_meta[cid] = (ab, rep_dist)
        if ab and rep_dist:
            cell_counts[(ab, rep_dist)] += 1

    for cid in sorted(clusters):
        cl = clusters[cid]
        is_linkable = (len(cl.members) >= 2) or _has_strong_evidence(clues_list[cl.members[0]])
        if not is_linkable:
            continue

        # Canonical sorted patient_key independent of insertion/set order
        anchor_parts = (
            sorted(cl.aadhaars)
            or sorted(cl.abhas)
            or sorted(cl.uhids)
            or sorted(cl.phones)
            or ["_".join(sorted("_".join(n) for n in cl.names))]
        )
        member_locs = sorted(f"{clues_list[m].file_rel}#{clues_list[m].loc}" for m in cl.members)
        patient_key = f"PT|{anchor_parts[0]}|{member_locs[0]}"
        pid = pseudonym(patient_key)
        shift = date_shift_days(patient_key, max_days=date_shift_max_days)

        # K-anonymity / re-identification risk check
        rep_age = sorted(cl.ages)[len(cl.ages) // 2] if cl.ages else None
        ab, rep_dist = cluster_meta.get(cid, (None, None))
        has_rare_diag = any(diag_counts[d] <= 2 for d in cl.diagnoses)
        small_cell = bool(ab and rep_dist and cell_counts[(ab, rep_dist)] < min_group_size)
        very_old = rep_age is not None and rep_age >= 80

        w = age_band_years
        suppress_dist = False
        if very_old:
            w = 20
            if rep_age >= 85 or has_rare_diag:
                suppress_dist = True
        elif small_cell and has_rare_diag:
            w = max(age_band_years, 10)
            if cell_counts[(ab, rep_dist)] <= 2:
                suppress_dist = True
        elif small_cell:
            w = max(age_band_years, 10)

        for m in cl.members:
            c = clues_list[m]
            c.patient_key = patient_key
            c.pseudonym = pid
            c.date_shift = shift
            c.age_width = w
            c.suppress_district = suppress_dist
            c.is_rare_combo = suppress_dist or (w > age_band_years)

    # For any unlinked records that still carry clinical dates, assign a deterministic per-record shift
    for c in clues_list:
        if c.patient_key is None:
            rec_key = f"REC|{c.file_rel}|{c.loc}"
            c.date_shift = date_shift_days(rec_key, max_days=date_shift_max_days)
