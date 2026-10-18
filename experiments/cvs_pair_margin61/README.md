# Fixed nominal practical one-sided margin tail

One AS_MARGIN row, seed 2026092701, inherits the exact P43 B01 As20 student and own EMA and adds 10 epochs. Total budget is E30 / 6660 updates; new work is 10 epochs / 2220 updates. All source RX, L6300 / hidden U56700 / V27000, nominal three-scene sampling, complete practical residual, original ordinary augmentation, old U rule, default cosine head, six L forwards and base loss scale 1.25 are retained.

The only added objective reuses two raw and two satellite logit forwards already present. It averages each pair, divides cosine logits by 30 and uses the detached clean-correct reference relation. The one-sided hinge has lambda .5, rho .5, clamp maximum 1. The fixed urban-only continuous added-noise weight affects this extra constraint; all legal labeled satellite samples keep their original CE. No new model forward, branch, head, U hook, curriculum or physical disturbance is added.

The source rule is fixed: full V clean versus reused NoAs30 >= −0.1 absolute pp; pooled panel mid/low versus reused As30 >= +0.5 pp; low >= As30. SOURCE_READOUT_DECIDED and SOURCE_READOUT_PARTIAL are stored before formal collection. Scientific FAIL and source-readout errors both retain the automatic seven-view prediction and independent truth-last scoring. The dispatcher then executes analysis/cvs_pair_margin61_analysis_20261018.py with --project and --out BASE/analysis_r01. Existing P43 NoAs30 / As30 predictions and scores are reused, never retrained or rescored. Failure does not authorize a sweep, seed expansion or E50.

The launcher's first step runs focused zero-NN checks and one actual As20 parent no-query smoke (2 NN / 1 checkpoint / 0 real IQ / 0 updates), then continues training. It never repeats the local actual F synthetic probe. GPU capacity is two total experiment PIDs including other owners and reservations before first CUDA use.

Local dependency checkout commands:

    python experiments/cvs_pair_margin61/local_synthetic_entry.py --dependencies PATH_TO_DEPENDENCY_CHECKOUT
    python experiments/cvs_pair_margin61/local_synthetic_entry.py --dependencies PATH_TO_DEPENDENCY_CHECKOUT --flow-only
    python experiments/cvs_pair_margin61/local_synthetic_entry.py --dependencies PATH_TO_DEPENDENCY_CHECKOUT --synthetic

The first two have zero CVS forward samples. The last is exactly one original six-by-128 synthetic forward/backward probe (768 NN, no real IQ/checkpoint/query/optimizer update), with explicit probe and process wall time in owner verification. Do not repeat it remotely.
