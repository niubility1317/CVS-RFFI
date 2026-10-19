# Complete HQ label-only E30

Run: 20261019-phase1-FcrossRX-signalBridge-e30-s2026092701-m3-r01. Three fixed own20+10 tails: N01 NoAs+extraRAW, C01 As+extraRAW and S01 As+extraHQ. F/CROSS_RX/default cosine, full source roles and old U are unchanged. Original 6L supervision remains absolute; append .25 times the mean of two CE64 forwards, with no extra KL, teacher, or cross-view constraint. All three have 896 L sample-forwards and nominal L scale 1.5 per update.

Extra HQ uses original raw L64 and fixed reference SNR 36 dB versus original 30 dB. Exogenous propagation/session hardware/standard-noise stream match; native quality, lock, compensation and AGC are recomputed. C01 generates identical HQ shadows. Fixed formal IQ and SNR do not change.

The old BRIDGE_CE included joint anchors, so it did not isolate pure supplemental label supervision. P70/P71 quality risk, U routing and P76 public propagation are closed and not rerun. This is one fixed test of observational supervision, not descriptor teaching or an Ag claim.

Launcher first covers actual own20 checkpoints, all three hook paths and final E30 boundary: 3592 sample-forwards, zero updates, no query. Every frozen row automatically completes seven predictions and independent truth-last scoring. Source-only criteria concern any future E50; failure still completes all7 and closes fixed +6 dB/.25 without parameter or seed rescue.

Entry: python -u experiments/cvs_signal_bridge30/runner.py. Local focused checks: experiments.cvs_signal_bridge30.checks.run().
