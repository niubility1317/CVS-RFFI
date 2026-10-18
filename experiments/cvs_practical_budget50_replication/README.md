# Fixed E50 seed02/03 replication

Four new rows extend their own fixed FINAL30 by 20 epochs: B00/B01 for model seeds 2026092702 and 2026092703. Seed01 is reused unchanged for all three-seed reports. This is the existing 20+10+20/reset AdamW/cosine20 recipe, not continuous cosine50 or a new augmentation mechanism.

The actual parent is the ordinary-account release `cvs_FcrossRX_As_replication_E30_s0203_20261018_r02`, commit `63b7c8cd7bb4139a2d800e6f4c4a7f1f2163fd47`. Its runner registers the real seed-aware factory and its pinned P43 dependencies. The new release archive includes this package and the unchanged `experiments/cvs_practical_budget50` package. The new namespace resolves unchanged channel, source-readout and prediction/scoring helpers from that bundled package.

The source loaders use the same L6300/U56700 hidden-TX/V27000 physical roles, all source RX/day, IQ256, default jointly trained cosine head and fixed practical processing. Native training and the F factory use each row's actual model seed. FINAL30 is local epoch10/EMA2220; the own scratch20 ancestor is epoch20/EMA4440. New training is local20/4440 and cumulative50/11100. Ordinary augmentation freezes own scratch20 epoch13; SupCon uses cumulative epoch+30.

Local focused checks (no Torch, checkpoint, IQ or target):

`python -m experiments.cvs_practical_budget50_replication.checks`

`python experiments/cvs_practical_budget50_replication/check_emitted.py experiments/cvs_practical_budget50_replication`

After registration and a committed package, place the ordinary-account private runtime in `local_artifacts/F_As_budget_E50_s0203_20261018/runtime.json`, then use `python -m experiments.cvs_practical_budget50_replication.publish`. Never overwrite/relaunch an existing release or output root.

Launcher smoke checks all four real parent lineages, performs one real-checkpoint no-query two-packet forward and four real-F synthetic 768-packet probes (shared 3074 NN, no optimizer update). It continues immediately on PASS. Four new rows each perform 4440 updates, 6271922 native NN, 12600 frozen source-panel NN and 1176002 fixed formal NN. Historical E30 and seed01 E50 costs remain separate.

Each row freezes FINAL50, performs source readout, predicts all seven fixed views, and enters a separate CPU-only truth-last scoring exec. Source scientific negatives do not suppress formal scoring. Reports retain all three seeds in fixed order and their paired mean/sample SD; formal results never change another row, seed, parameter or epoch. No automatic repetition or seed expansion is implemented.
