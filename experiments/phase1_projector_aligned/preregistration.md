# Independent feature projector with aligned source objective

Keep the existing frozen-feature projector architecture, frozen Tri-SIG-TIME SSL, blind estimator, gate, source sampling, data views, random seed, optimizer and five-epoch budget. Replace the original unaligned paired-IQ/feature primary objective with the existing aligned-IQ, feature, frozen-reader distribution, pseudo-margin and power objective. Retain the predicted-feature objective with weight 0.5. This is a comparison of the whole primary loss, not an isolated alignment-only ablation. Independently train from initialization.

Training uses paired Phase1 source IQ and frozen teacher features without TX identities, U_s truth, V or ground data. Inference uses one received IQ and fixed components and does not load DURA or any clean reference. Fix predictions before source L_s scoring and fix all seven ground views before independent evaluation. Source and ground development data have previously been explored; no blind-test or in-orbit claim.

Source 20261103-phase1-projector-aligned-source-s2026092701-r01. Ground 20261103-phase1-projector-aligned-ground-test-s2026092701-r01.
