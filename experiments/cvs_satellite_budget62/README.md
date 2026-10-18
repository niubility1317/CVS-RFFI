# Source-qualified satellite dropout: total E50

This is a budget extension of the pinned SAT18 AS_CE / AS_DET recipe, not a new augmentation or loss. Each seed 2026092701 row loads its own fixed FINAL30 student and EMA and trains 20 more epochs. Native counters are local E1–20 / 4440 updates; channel, SupCon and ordinary scheduling use the declared cumulative E31–50 / 6661–11100 context. AdamW / cosine20 and temporal cache reset; ordinary augmentation remains frozen at E13 using that row's own scratch20 arguments.

No new head, branch, U route, Ag, alignment, margin, teacher or deployment module is added. Existing original/default cosine head remains jointly trained. Full practical residual / post_sync / EQ-off G=1 and all formal arrays remain unchanged. All legal labeled satellite samples receive CE.

AS_CE keeps original stochastic dropout and removes satellite-half KL. AS_DET retains the same objective but bypasses only existing dropout outputs on the actual 64 satellite rows, still drawing native full masks. Both use the pinned SAT18 implementation. Ordinary/raw/U dropout and RDrop remain unchanged; each update has six L forwards / 768 samples and total L scale 1.25.

The two new rows and six reused rows are returned by design.logical_rows(). Existing predictions, scores and source readouts are reused; they are never retrained or rescored. New rows always predict clean plus all six practical views and use a separate truth-last scorer, even when source checks fail. The eight-row CPU analysis is executed automatically after closure. Source decision is fixed before formal collection and never reads formal scores; incomplete source readout is preserved as PARTIAL without blocking legal formal closure.

The only E50 scientific hypothesis is whether additional budget retains or improves the source-supported satellite dropout benefit. Source checks: DET50 versus matched NoAs50 clean no worse than −0.1 absolute pp; DET50 versus CE50 pooled mid/low at least +0.5 pp; low not worse. A failure does not trigger a sweep, retry, E70 or seed expansion.

Prepare runtime.json from runtime.example.json with actual pinned release paths. The immutable release contains this package and analysis/cvs_satellite_budget62_analysis_20261018.py. Run runner.py once; its first step is one compound no-query smoke of both actual FINAL30 parents (4 NN / 2 checkpoints / 0 real IQ / 0 updates), immediately followed by dispatch. Global GPU cap is two total experiment PIDs including other owners.

Focused zero-forward checks:

    python -m experiments.cvs_satellite_budget62.checks

Local-only dependency checkout checks (no IQ/checkpoints/target):

    python experiments/cvs_satellite_budget62/local_synthetic_entry.py --dependencies PATH_TO_DEPENDENCY_CHECKOUT --flow-only
    python experiments/cvs_satellite_budget62/local_synthetic_entry.py --dependencies PATH_TO_DEPENDENCY_CHECKOUT

The second command is two actual F synthetic forward/backward probes, 1536 forward samples total, with no optimizer update. It also checks native full-mask RNG, ordinary logits, default-head and Sinc gradients, own-EMA initialization, empty reset AdamW, and scratch20 E13 ordinary schedule. Do not rerun it on the remote dispatcher; remote smoke is exactly 4 NN.
