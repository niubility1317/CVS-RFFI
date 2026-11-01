# Phase1 source common-frame DURA action selector

Source-only diagnostic. Frozen Tri-SIG-TIME SSL and DURA are used. Eight DURA inputs use global phase rotations. The original selector inversely rotated outputs; the corrected, posthoc selector keeps all output waveforms in the observed common phase frame. Each selects the lowest DURA internal forward-residual / estimated variance subject to four fixed relative-improvement thresholds and a previously developed blind noise gate. The internal score is not a true channel likelihood. Inference requires received IQ and fixed models only.

Using 1,051 source validation and 564 source receiver-8 holdout records (with confident U_s teacher pseudo-targets and true L_s labels), the original-coordinate selector changed aggregate validation correct counts by -157, -138, -97, -26. Correcting the frame yielded -4, -2, -7, 0 respectively; a pre-existing selection rule retained DURA, so no heldout action selector was evaluated. The corrected complete run was repeated and its JSON matched exactly. This is posthoc on source validation records already inspected.

In a separate 271-valid-target source holdout from an already examined sample, the label-assisted common-frame phase orbit gave low-urban 242 baseline, 251 oracle, or 246 when repaired candidates additionally had to improve paired clean aligned IQ error. True L_s baseline 63/69, fidelity oracle 65/69. These per-record oracles use unavailable labels/paired clean IQ and are not deployable; they concern these fixed candidates only.

No U_s true labels, validation IQ or test IQ were read. Frozen models saw source records; source IDs may overlap other diagnostics. Simulated satellite data has not been validated as a real link. No target test was run.
