# Exact within-step scene counts: originalP43 As30

One fixed new row B01-s2026092701 loads its lawful own originalP43 AsFINAL20 student and EMA,then trains10 additional epochs/2220 updates,totalE30/6660. OriginalNoAs30 andAs30 frozen readouts are reused. No newNoAs row: the scene operator is an identity when As is absent and uses no global RNG.

Current P43 already chooses each packet's scene independently among low_suburban,mid_urban,low_urban. This candidate only replaces random perstep scene counts by21/21/22. Extra scene index=(cumulative_step-1)%3,so step4441 gives low_suburban22,4442 mid_urban22,4443 low_urban22. An independent named default_rng permutation assigns the balanced scene vector to the original64 input positions. Each three steps has64 observations per scene,each222step epoch4736,and the tail10 epochs47360 per scene. Original sampler positions and duplicate physical IDs remain;no deduplication/uniqueness constraint or label-based assignment.

The new adapter copies the actual original downstream lines: choose cfg,stable_seed(train_seed,namespace,assigned_scene,pid),receiver_for_session,normalize_input,ResidualChannel.process and finite/noEQ checks. Session hardware,geometry-dependent state priors,noise,quality-dependent lock/compensation,AGC,NoExplicitCFO/internal compensation and formal IQ remain unchanged. Repeated pid with the same assigned scene in one step retains the original same seed;the per-position allocation can assign repeated pid appearances to different scenes. This is intentional position stratification,not a change to the physical sampler. The old perphysical six-scene visit-cycle differs from the exact three-scene within-batch allocation;this does not claim that multi-scene augmentation is new.

The model delegates directly to originalP43 model.hook,ORD128 SupCon and its actual rawNONE control pair. It does not inherit MIXED128 input contrast or weighted .125/.375 losses. Original .5 RDordinary128+.5 RDmixed128+.25 RDraw128+.1 ORD128CROSS_RX remains:6L/768 neural packets per update,L scale1.25. Native optimizer epoch10/update2220 and cumulative30/6660 are explicit. Ownoriginal20 augmentation args freeze at epoch13;SupCon+20,channel+20/+4440,AdamW reset/cosine10,defaultjointcos head,F architecture,oldU and ownEMA stay fixed.

Fullsource L6300/U56700 hiddenTX/V27000,all5RX/all3days/IQ256/split392005 remain. New training L neural packets1704960;As142080;no additional mechanism NN or U views. Parent20 and original30 costs are reused,not charged as new.

Launcher first step: one actual own20 no-query CP readout2NN plus one real6L synthetic128 probe768NN,total770NN/0 realIQ/0 updates. The actual default head,Sinc/backbone gradients,original ORD128 gradient path,6L budget and first balanced64 call are checked. PASS proceeds directly to normal training;local checks never claim this realCP smoke already ran.

Each fixedFINAL30 automatically proceeds through source readout,clean+all6 fixed practical predictions,then separateexec truth-last scoring. Source eligibility has exactly two criteria:newAs-fullV originalNoAs30 clean>=-.1 absolutepp,and two-draw mid/low mean newAs-originalAs30>=+.5pp. Failure closes this fixed recipe;all7 formal still finish;no automaticE50/seed/phase rescue. The formal clean stage compares newAs to the reused matching originalNoAs30 at>=-.1pp.

Producer per-row paths:
- source/reference_steps.jsonl.gz: original common/extra RD/CE/KL,SupCon/details,scene_counts/state_counts,ordinary_raw_MSE/second_raw_MSE;total_L is full6L.
- source/shape_steps.jsonl.gz: unchanged actual raw pair RD/CE/KL,weight.25,sameIQ and cumulative step.
- source/scene_balance_steps.jsonl.gz: each actual original64 input position,assigned scene/private permutation/seed,physical metadata and trueL TX/RX/day,state/lock,unchanged downstream channel seed,original selectedL positions. Existing training_sequence remains the L/U physical sampler trace.
- source/unit_steps.jsonl.gz,result.json,completion.json,source_frozen.json,actual_modules.json,actual_channel.json and ordinary_augmentation_actual.json preserve native/ownEMA/optimizer/source costs.
- source_qualification/{predictions.npz,complete.json};prediction/{sevenview_scores.npz,predictions.npz,complete.json};scoring/{scores.json,complete.json}.

Native14 metrics=schema.py CONCAT_FIELDS7+BALANCE_FIELDS7. There is no shape/RD metric in this new wrapper and no selected_actual_view/selected_MSE invented in original reference_steps. Raw terms are in shape_steps. Contrast remains original SupCon_details in reference_steps;there is no added contrast_steps or weight_steps.

stdout=run/logs/B01-s2026092701.log;logical report rows=NOAS30,AS30,BAL_AS30. Launcher/runtime require pinned originalP43 release19aedc3604fed9c193237dd92366164178af0983. No target-selected or unknown parent fallback. Publish only through the parent owner's one focused review/commit/preregistration and ordinary-account minimal workflow.
