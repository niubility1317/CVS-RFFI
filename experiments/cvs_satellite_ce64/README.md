# Satellite-only CE temperature, fixed E30

New rows CE_HALF and CE_T2 both strictly inherit the same original P43 own As20 student and EMA and reset AdamW/cosine for ten epochs (total E30). All five source RX, complete L6300/U56700 hidden/V27000, original six-view labeled execution, native U, RDrop, physical channel and deployment cosine30 head remain.

Only satellite64 CE changes: CE_HALF = 0.5 CE(z,y), CE_T2 = CE(z/2,y), label smoothing .01, no T² multiplier. Each pass CE is computed separately and averaged. As the two mixed128 CE terms have total outer weight .5 and satellite64 occupies half, correction is .25*(new_pair_CE64-old_pair_CE64). Non-satellite CE and all symmetric KL terms are untouched. The explicit derivative prefactor is 1/2 in both arms; actual parameter gradient norms are not matched.

Source existing logits do not support highly confident urban errors. This is a falsification of probability-shape gradient redistribution, not a claim to correct a particular physical impairment. Reuse P43 As30 and NoAs30 frozen comparisons; no formal score may change temperature, rows or promotion.

Run: python -u experiments/cvs_satellite_ce64/runner.py

First step focused checks and one actual ownAs20 no-query smoke; then two independent rows train, fixed final checkpoints are read on legal source V, and each row immediately predicts all seven fixed formal views before independent truth-last scoring. Source qualification failure still completes every formal view. One immutable run root, no automatic retry/E50/seed expansion.

CPU final analysis: analysis/cvs_satellite_ce64_analysis_20261018.py. Costs, full epoch/step curves, source state/lock/added-noise strata, formal RX/day/TX metrics and repaired/new errors are saved; two channel draws are not two model seeds.
