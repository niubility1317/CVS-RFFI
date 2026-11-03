# Local complex projection teacher for DURA/physical fusion

Retain the wide physical-fusion network, frozen DURA and physical posterior, source records, views, random seeds, optimizer and five-epoch budget. Add a source-only auxiliary loss with weight 0.25: align paired clean IQ and physical posterior to DURA, derive a bounded complex coefficient from a 17-sample local least-squares projection, and train the resulting candidate toward that projection. This is a local waveform supervision proposal, not a classification upper bound. The existing protected objective and source pseudo-label rescue term remain unchanged.

Clean IQ and frozen-reader source features are training-only. No TX identity, U_s truth, V or ground data are read during training or prediction; inference uses one received IQ, frozen components and the learned model. Fix source predictions before source L_s scoring, and fix all seven Phase1 ground predictions before ground truth scoring and independent audit. Source and ground development records were explored before and are not independent blind or in-orbit data.

Source: 20261103-phase1-dura-physical-local-teacher-source-s2026092701-r01. Ground: 20261103-phase1-dura-physical-local-teacher-ground-test-s2026092701-r01. Compare to the independently trained wide variant, but different objectives cause different optimization paths.
