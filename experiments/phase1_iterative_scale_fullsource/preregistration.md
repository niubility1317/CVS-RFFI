# Matched source replay of iterative recovery across fit sizes

Compare independently trained 8,192-record and 32,768-record standalone iterative time/frequency restorers on the same 11,761 explored Phase1 source holdout records. The larger fit is the exact deterministic superset of the smaller fit, excludes the same physical-prior records and held-out records, and uses the same model, three source views, optimizer, seed and five epochs. Data volume and optimization-step count both increase; results are not a pure sample-coverage ablation. Neither restorer loads DURA.

Require both fixed source scores and independent audits, regenerate and verify the exact registered nested fit selections, compare raw predictions and blind gate, and reproduce both 672-record fixed pilots. Freeze all 11,761 predictions and hashes before connecting 1,047 authorized L_s identities and frozen-reader pseudo classes. Independently recount 16 groups, 96 comparisons and 480 fields. Do not read U_s truth, V or ground data. Previously explored source records are not blind or in-orbit evidence; seven-view ground tests are separate.

Run 20261103-phase1-iterative-scale-fullsource-s2026092701-r01.
