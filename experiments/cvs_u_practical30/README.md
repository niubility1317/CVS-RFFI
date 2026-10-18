# Fixed teacher-only qualification / quarter U practical E30

The r02 run uses the fixed N01/C01/S01 matrix. The original m2-r01 run is preserved as a technical failure before any worker/update. N uses strict original P43 ownNoAs20+EMA; C/S use the same strict original P43 ownAs20+EMA. All append ten epochs with native resetAdamW/cos10. C/S difference is only the U student input route; N is the matched teacher-only no-L-As control.

Both C/S and N set native pseudo_strong_agreement=False, preserving original raw clean EMA teacher quantile and batch_neighbor qualification. Native strong Gaussian .015 runs once on every U packet before S overwrites a private uniform quarter of positions using complete practical of original raw U. C and N keep Gaussian in all slots. No U TX truth is exposed. The pseudo CE still averages across qualified U slots; entropy averages over all U slots including unqualified/practical. Student trajectories and teacher qualification outcomes are not claimed equal after training.

Original P43 L path delegates unchanged: 6L/768 NN, .5 ordinary RD + .5 mixed RD + .25 raw RD + .1 ORD128 SupCon. All RD pairs share the already-generated IQ. N has raw64 in selected L positions; C/S have completeAs(raw64). No extra NN is introduced in actual training. S adds 141750 U propagation simulations over the ten epochs, besides original 142080 L As.

The real frozen native interface is model._source_unlabeled_feature_hook. A scoped logger observes the native pseudo_loss/entropy_loss and returns (None, {}), so the native loss/node remains unchanged. No module-level trusted pseudo hook is created.

Artifacts: source/training_sequence.jsonl.gz; reference_steps/unit_steps/shape_steps/u_steps.jsonl.gz; actual_U_channel.json; ordinary_augmentation_actual.json; EMA_initialization.json; result.json. U stream logs teacher confidence/pseudo/masks, student logits, native loss and denominators, route positions/physical metadata/physics and CPU timing. Native step metrics only emit seven concat fields from the actual full6L record.

Launcher first runs unique-parent no-query + actual three-row 6L and synthetic U route probes, then trains on PASS. All fixed FINAL30 models complete clean+six practical prediction and independent truth-last score even if source science criteria fail. Scientific criteria are S-N fullV clean >= -.1 absolute pp and S-C mid/low panel mean >= +.5 pp; no automatic E50 or seed expansion.
