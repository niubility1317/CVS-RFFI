# LRaw in the final ten epochs

Two fixed B00/B01 seed2026092701 rows each load the corresponding original P43 scratchFINAL20 student and ownEMA,then continue10 epochs/2220updates,totalE30/6660. OriginalNoAs30/As30 readouts are reused. The first20 epochs retain ordinary augmentation;this is not whole-trajectory NoAug.

The unique native ordinary L augmentor factory is wrapped. Each actual L128 call with visible labels0..5 and the original labels/no_paFalse signature saves the pre-call raw clone,executes the full original augmentor once on an independent clone,discards its generated IQ and returns the saved raw. The original raw input cannot be mutated by that call. Schedule and every original augmentation operation still execute;no augmentation compute saving is claimed. U weak/strong functions and teacher calls are not wrapped. Local interface checks compare same-call Python/NumPy/RNG-double endstates;actual same-call Torch RNG is checked in launcher smoke. Full-training RNG equality is not claimed.

This differs from old PA-only bypass,which returned the post-general-channel prefix. It removes the full ordinary signal transformation from the returned L ordinary slot. Original As already uses raw IQ,so this is not a repair of duplicated ordinary/As damage. NoAs still means only As-off;its original ordinary slot was augmented.

The original P43 hook is delegated: .5RDordinarySlot128+.5RDmixed128+.25RDraw128+.1ORD128 CROSS_RX,6L/768NN eachstep,Lscale1.25. In the final10 ordinarySlot128 is raw128,and the ordinary64 half of mixed128 is raw too. NoAs selected64 remainsraw;As selected64 remains one complete practical(raw64). RD passes reuse their already generated IQ. ORD128 remains the original map positions/positive-negative matrix/schedule,but its actual ordinary anchor input is nowraw128. No new loss,layer,coefficient,head or deployment structure.

OriginalU/ownEMA,use_aug=True,phase0/10,hiddenU labels and defaultjoint6x160 cosine head remain. Ordinary scheduling uses strict own20 args/base configuration frozen13;SupCon+20 and channel+20/+4440 retain cumulativeE21..30. Reset AdamW/cosine10,F900722 parameters/206 state keys,allsource L6300/U56700/V27000,all5RX/day1..3/IQ256/split392005 remain. Each row adds1704960 L neural packets and567000 existing U student packets;ownteacher evaluation567000,teacher training0. B01 As142080/B00 As0. Teacher/source/eval/wall/smoke costs remain separate.

Launcher first step:two actual own20 parent no-query readouts2NN each plus two actual6L synthetic128 probes768NN each,total1540NN/0 realIQ/0 updates. Each probe uses the actual wrapped factory and original epoch13 augmentation. A same-input reference ordinary generation is used only to verify local RNG/MSE;smoke records512 synthetic ordinary-generation packets across both rows. It checks defaulthead/Sinc/backbone gradients,ORD128 map routing,6L1.25,As64/0,same-IQ RD and original enabledU/EMA flags. PASS directly trains. Real smoke has not been run by local no-Torch checks.

Each fixedFINAL30 automatically completes source readout,clean+all6 fixed practical predictions,and independentexec truth-last scoring. The only source eligibility rules:newAs-newNoAs fullVclean>=-.1 absolutepp and newAs-originalAs two-drawmidlowmean>=+.5pp. A scientific FAIL still finishes all7 formal views,closes this fixed recipe and causes no automatic E50/switch/phase/seed rescue. Formalstage clean compares matched newAs-newNoAs>=-.1pp. One seed cannot prove repeatable dual improvement.

Logical rows:NOAS30,AS30,LRAW_NOAS30,LRAW_AS30. Per new physical row:
- source/reference_steps.jsonl.gz:original RD/CE/KL,SupCon/details,scene/state counts and raw MSEs;total_L is full6L and ordinary_raw_MSE is0.
- source/shape_steps.jsonl.gz:unchanged .25 raw RD/CE/KL and same-IQ records.
- source/lraw_steps.jsonl.gz:seven LRAW_FIELDS plus native/cumulative epoch/step,actual raw anchor input and same-IQ flag. Nativeinfo=concat7+lraw7;no shape/RD invented.
- source/ordinary_draw_steps.jsonl.gz:each wrapper call count128,input clone/full native draw evidence,discardedordinary_MSE/changed count and returned_raw_MSE0. DiscardedMSE is logged finite/nonnegative,not a candidate identity gate.
- source/training_sequence.jsonl.gz retains real L and fullU traversals. result.lraw aggregates factory/calls/packets/MSE and explicit first20 history/RNG limits;EMA_initialization.json restores strict ownstage20 EMA,result reports real teacher counters and2220local/6660cumulativeEMA updates. ordinary_augmentation_actual.json records parent20base/frozen13/fullcompute/rawreturn/Uunwrapped.
- source_qualification/{predictions.npz,complete.json};prediction/{sevenview_scores.npz,predictions.npz,complete.json};scoring/{scores.json,complete.json}.

stdout=run/logs/{B00,B01}-s2026092701.log. Runtime requires pinned originalP43 release19aedc3604fed9c193237dd92366164178af0983;no unknown or target-selected fallback. Local checks do not load Torch/checkpoints/IQ or start NN/remote operations.
