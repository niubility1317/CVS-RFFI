# Phase1 privileged source convex-mixture opportunity

Status: privileged source-only exploratory analysis; not a deployable model, a target-test measurement or a real-satellite result. The idea follows previously observed Phase1 test expert-oracle results, so later use of that test is not blind.

The experiment uses 256 fresh source physical IDs, comprising 64 L_s and 192 U_s, and independently seeded practical residual-channel simulations in three scenes. Of these, 251 records per view are valid: 64 with true L_s labels and 187 with confident clean-teacher U_s pseudo-labels. The frozen classifier is Tri-SIG-TIME SSL, and the four input signals are raw, DURA, boundary and feature-conditioned. Those fixed experts previously trained on source records. All expert outputs were fixed before source labels were joined.

For each individual source record, Adam performs 30 steps on four softmax mixture weights to maximize the source target's classifier likelihood. The method retains any successful iterate using the target and falls back to DURA when no iterate succeeds; this target-dependent choice is unavailable at deployment. The convex search is not exhaustive and does not necessarily include each individual expert exactly. Therefore its observed count is a search result, not a universal mixture upper bound. The four-expert oracle similarly knows the target and is restricted to the four fixed predictions.

| Source view | DURA correct | Four-expert oracle | Convex search correct | Convex-only correct beyond all four experts | L_s true-label correct: DURA → search |
| --- | ---: | ---: | ---: | ---: | ---: |
| practical_low_suburban | 245 | 250 | 250 | 1 | 60 → 63 |
| practical_mid_urban | 229 | 247 | 246 | 1 | 58 → 63 |
| practical_low_urban | 217 | 242 | 242 | 4 | 56 → 62 |

The low-urban convex search finds four correct decisions that none of the four individual predictions made, and increases correct mixed-target observations from 217 to 242. This supports investigating an amortized mixture network, but does not establish that a network can learn those weights or generalize. Simulation is not a validated satellite channel. No U_s transmitter labels, validation IQ or test IQ were read; the script and JSON retain physical seeds and accounting. The run was independently repeated with identical results.
