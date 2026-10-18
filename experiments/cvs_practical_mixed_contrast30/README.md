# Existing mixed128 SupCon membership (E30)

One fixed seed, two new rows: B00/B01-s2026092701. Each loads its own original P43 FINAL20 student and EMA, then runs exactly10 additional epochs (2220 updates), ending at cumulativeE30/6660. Original E30 NoAs/As readouts are reused; no original20 or original30 retraining.

The sole change is which existing maps feed the existing CROSS_RX SupCon: MIXED128 instead of ORD128. The128 original batch positions each contribute one anchor per term, preserving their order and labels/RX. Existing sampler duplicates are neither rejected nor removed. This excludes SAT192's extra64 duplicated positions; it does not claim all128 sampled physical IDs are unique. SAT192 already failed its source criterion, and its failure is retained.

The network call sequence remains ordinary pass0/pass1, mixed pass0/pass1, raw pass0/pass1:6L=768 neural packets per step. The objective remains .5 RDordinary128+.5 RDmixed128+.25 RDraw128+.1 SupCon, supervisedL scale1.25. SupCon takes the corresponding existing mixed pass maps at time_down/t2.pool, with unchanged descriptor, normalization, temperature.2 and relation matrix. It adds no loss, layer, IQ draw, NN forward or deployment branch. NoAs selected64 remains original raw64; As selected64 remains completepractical(raw64). The original64 mask, fullphysics, normalization, NoExplicitCFO, internal packet compensation, U rules, ownEMA, joint defaultcos6x160 head and F architecture are preserved. Ordinaryaugmentation uses ownoriginal20 args frozen at epoch13.

Each new row contributes1704960 L neural packets; As contributes142080 actualAs packets and NoAs0. OriginalL6300/U56700 with hiddenTX/V27000, all5RX and all3days/IQ256 remain unchanged. Native optimizer counters remain10/2220; cumulative metadata is30/6660.

Launcher first step: strict two real own20 CP no-query readouts (2NN each), plus two actual6L synthetic128 probes (768NN each): total1540NN,0 realIQ/updates. The same-map ORD128 formula/gradient comparison uses tensor algebra and0NN. PASS continues directly to training; local pure checks do not claim this real smoke has run.

After fixedFINAL30: source readout, all7 fixed predictions, then separateexec truth-last scoring automatically complete each row. Scientific sourceFAIL does not skip formal testing. Source future eligibility only: newAs-originalAs fullVclean>=-.1 absolutepp and V1800 two-draw mid/lowurban mean>=+.5pp. Formal stage target: newAs-matchednewNoAs clean>=-.1pp. No automaticE50/seed extension.

Producer paths per row: source/{reference_steps,shape_steps,contrast_steps,unit_steps,training_sequence}.jsonl.gz and source/result.json; source_qualification/{predictions.npz,complete.json}; prediction/{sevenview_scores.npz,predictions.npz,complete.json}; scoring/{scores.json,complete.json}. Stdout is run/logs/{physicalrow}.log.

Native14 metrics are declared in schema.py. concat/total_L means full6L. RawRD is in shape_steps, never an invented shape/RD metric. contrast_steps records original128 positions/trueL metadata, selected64, actualAs count/view, four correspondingphase/level mapshapes and0 extraNN/rawmaps. Existing training_sequence supplies originalphysical metadata; no fullIQ-equality claim follows from map/sequence logs.

Private runtime.json must point to the pinned P43 release commit19aedc3604fed9c193237dd92366164178af0983; no weighted or target-selected initializer is accepted. Publish only after the parent task's one focused review/commit/preregistration, using the ordinary account, one archive comparison/compile, immutable outputs and capacity3/free12000MB.
