"""Build a candidate model inventory; this is not an experiment launcher."""
from collections import Counter
import csv
from pathlib import Path

OUT = Path(__file__).resolve().parent / "candidate_models.csv"
SEEDS = (2026092701, 2026092702, 2026092703)
SETTINGS = ("main", "receiver_1", "receiver_2", "ten_class", "label_1pct",
            "label_5pct", "sources_1", "sources_2", "sources_3", "sources_4", "sources_6")
LEARNING = ("sup_aug", "pl_aug", "sup_noaug", "pl_noaug", "full_ssl")
STRUCTURES = ("complete", "single_view", "no_time_coordinate", "no_signature",
              "no_pa", "moment_pooling", "fixed_permutation", "no_time_backbone",
              "no_frequency_backbone")
rows = []


def add(group, setting, method, structure, seed, epochs, reuse="", evidence=""):
    rows.append(dict(group=group, setting=setting, method=method, structure=structure,
                     model_seed=seed, epochs=epochs,
                     candidate_status="CANDIDATE_REUSE" if reuse else "CANDIDATE_NEW",
                     reuse_row=reuse, evidence=evidence))


for setting in SETTINGS:
    for method in LEARNING:
        for seed in SEEDS:
            reuse = f"main_split-structure-ssl-no_reference-permuted-s{seed}" if setting == "main" and method == "full_ssl" else ""
            evidence = "20261027-phase1-pamf-paper-completion-m119-r03" if reuse else ""
            add("main_and_factorial", setting, method, "three_view_no_reference", seed, 50, reuse, evidence)

for seed in range(2026092704, 2026092709):
    add("main_eight_seed_extension", "main", "full_ssl", "three_view_no_reference", seed, 50)
for seed in range(2026092701, 2026092709):
    add("architecture_e200", "main", "ce_no_augmentation", "three_view_no_reference", seed, 200)
for structure in STRUCTURES:
    for seed in SEEDS:
        reuse = evidence = ""
        if structure == "complete":
            reuse, evidence = f"F0-CE_NO_REFERENCE-s{seed}", "20261024-phase1-PAMF-REPL-completion-m201-r01"
        elif structure == "single_view":
            reuse = f"main_split-branch-no_reference-supervised-channel_aug-s{seed}"
            evidence = "20261027-phase1-pamf-single-view-paper-m120-r01"
        elif structure == "no_signature":
            reuse, evidence = f"NO_REFERENCE-s{seed}", "20261026-phase1-PAMF-noSIG-supervised-ablation-m15-r02"
        add("structure_ce", "main", "ce_aug", structure, seed, 50, reuse, evidence)

counts = Counter(row["candidate_status"] for row in rows)
assert len(rows) == 205 and counts == {"CANDIDATE_NEW": 193, "CANDIDATE_REUSE": 12}
assert len({(r["group"], r["setting"], r["method"], r["structure"], r["model_seed"]) for r in rows}) == 205
with OUT.open("w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=rows[0])
    writer.writeheader()
    writer.writerows(rows)
print(f"{OUT}: {len(rows)} logical rows; {counts['CANDIDATE_NEW']} candidate new; {counts['CANDIDATE_REUSE']} candidate reuse")
