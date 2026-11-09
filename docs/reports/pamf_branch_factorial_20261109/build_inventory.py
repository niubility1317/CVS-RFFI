"""Produce a draft model inventory; this file does not launch experiments."""
from collections import Counter
import csv
from pathlib import Path

here = Path(__file__).resolve().parent
old = here.parent / "pamf_no_reference_inventory_20261109" / "candidate_models.csv"
with old.open(encoding="utf-8", newline="") as f:
    rows = list(csv.DictReader(f))
assert len(rows) == 205

extras = (
    "strict_no_frequency", "time_only", "frequency_only", "pa_only",
    "no_signature_no_pa", "no_signature_strict_no_frequency", "no_signature_time_only",
)
for structure in extras:
    for seed in (2026092701, 2026092702, 2026092703):
        reuse = f"NO_PA_REFERENCE-s{seed}" if structure == "no_signature_no_pa" else ""
        rows.append(dict(group="structure_ce", setting="main", method="ce_aug", structure=structure,
                         model_seed=str(seed), epochs="50",
                         candidate_status="CANDIDATE_REUSE" if reuse else "CANDIDATE_NEW",
                         reuse_row=reuse,
                         evidence="20261026-phase1-PAMF-noSIG-supervised-ablation-m15-r02" if reuse else ""))

keys = [(r["group"], r["setting"], r["method"], r["structure"], r["model_seed"]) for r in rows]
counts = Counter(r["candidate_status"] for r in rows)
assert len(keys) == len(set(keys)) == 226
assert counts == {"CANDIDATE_NEW": 211, "CANDIDATE_REUSE": 15}
assert Counter(r["epochs"] for r in rows if r["candidate_status"] == "CANDIDATE_NEW") == {"50": 203, "200": 8}
out = here / "candidate_models.csv"
with out.open("w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)
print(f"{out}: 226 logical models; 211 candidate new; 15 candidate reuse")
