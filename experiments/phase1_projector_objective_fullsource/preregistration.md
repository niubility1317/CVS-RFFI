# Matched source replay of independent projector objectives

Replay the fixed original and aligned-objective feature-projector models on the same 11,761 explored Phase1 source holdout records. Both use the same architecture, ordered 8,192 fit IDs, three view cache, seed, optimizer and five epochs. The independently trained primary objectives differ in alignment, distillation, margin and power terms, so this is not an alignment-only ablation.

Verify both embedded frozen-reader states against the current source reader, raw decisions and gate against the fixed comparator, and reproduce both 672-record fixed pilots. Freeze all new predictions and hashes before scoring 1,047 allowed L_s identities and clean-reader pseudo classes; independently recount 16 groups, 96 comparisons and 480 fields. Read no U_s truth, V or ground data. Previously explored source records are not blind or in-orbit evidence; ground evaluations remain separate.

Run: 20261103-phase1-projector-objective-fullsource-s2026092701-r01.
