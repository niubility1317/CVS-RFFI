# Raw-observation conditioned IQ flow

DURA-free conditional velocity field starts at the received IQ and integrates a four-step deterministic flow. Phase-invariant local features predict complex coefficients applied to observation, current state, and learned complex temporal filters. It is globally phase equivariant; zero initialization is identity, and no channel estimator, candidate router, or DURA is loaded.

Source pilot uses 4096 source training physical IDs, three pre-existing authorized impaired/clean views (epochs 21–23), and paired flow matching on straight interpolations. The network uses no TX labels, U_s identity, validation V or ground test; frozen Tri-SIG-TIME SSL is used only for source evaluation after training. Initial rollout uses four fixed Euler steps. Source RX4/6 and RX8 are previously explored and F has seen source training, so pseudo-label agreement is not independent ground accuracy. No automatic ground prediction is authorized by this pilot.

Local tests verify exact identity initialization, global phase equivariance after nonzero weights, finite learning gradient, contextual signal reconstruction and invalid step rejection. A 100-step synthetic paired scaling example reached loss 0.09227 → 0.0000607. These are implementation checks only.
