# Phase2 practical residual baseline execution

The user confirmed `residual_noeq` for Phase1 and every Phase2 support/query observation: `processing_route=residual`, `mode=post_sync`, `equalization_enabled=false`, sampling rate25MHz. Input WiSig equalized1 is an existing dataset representation; the new channel applies no additional equalizer.

## Scope and fixed comparisons

Phase1 source run: `20260927-phase1-baselines-practical-manysig-m5-r01`, source release `d8d0f333d968a7ec4df3b85636c474c634c6c8cb`. Forty scratch-trained rows cover CVCNN-CE, RIEI-FD and DRIFT, each with CE and PL, plus POSTER and RadioNet DF. All use200epochs and the same source physical roles. Final epoch200 is used regardless of validation accuracy.

Model seeds are392005 and2026092701–2026092704 for every method.392005 is historically optimized and must be reported separately from the four fresh seeds. No target-based seed selection, selective omission, or performance-based rerunning is permitted. These seeds do not establish CVS superiority; the resulting measurements determine that claim.

Phase2 data run: `20260927-phase2-practical-data-manytx-s2026092705-r01`, builder release `7dd3e79a62c8f7e8487ceeff2dd9ae85ca1e74ee`. The completed capsule is `residual-noeq-ba667eee4fb061055e4c08b5`, with36,036fixed observations and2,100splits. Seven target receivers are physically disjoint from all five source receivers. Both old and new target IQ come from ManyTx. This study has6old classes and up to20new classes, not20old classes.

For each receiver/class,198distinct physical records supply three disjoint scenes, each with36support-pool records and30fixed query records. K=1,5,10,20 is nested within five support draws2026092711–2026092715. New-class counts are0,2,5,10,20. All methods reuse the same received IQ and split IDs.

## Prediction routes

Phase2 run: `20260927-phase2-baselines-practical-manytx-m5-r01`.

- Frozen DG: unchanged six-class source classifier,21receiver/scene combinations per source model. No support adaptation.
- Support NCM: unit-normalized frozen embeddings, mean per registered class, cosine scores over every registered class. This is a common registration extension of each source representation, not a claim to reproduce an original new-class algorithm.
- POSTER Fine-Tuning: author-code convention replaces all three Dense layers, freezes convolutional weights, trains10epochs withAdam0.001/epsilon1e-7/batch256 on support only.
- RadioNet: DF backbone, replace classifier head, freeze convolutional weights, train30epochs withAdam0.001/epsilon1e-7/batch128 on support only. RadioNet ADA/triplet variants are outside this batch.

The source budget200epochs is a matched project variant; it differs from the original author source budgets. The PyTorch ports were numerically checked against author Keras architectures on CPU, with maximum absolute output discrepancy1.49e-8. Dropout training mode and author freeze boundaries are retained during supervised adaptation.

## Execution and evidence

Each Phase2 GPU lane waits for its own registered source row's completion. It validates scratch provenance, exact source physical roles, fixed200epoch completion, and no previous target evaluation before loading `last.pt`. A real-checkpoint support-only smoke precedes query inference. One source lane plus one Phase2 lane gives at most two training processes/GPU. Technical failures stop the affected lane without deleting outputs or retrying.

Query inference is evaluation-only and per sample; it never estimates batch statistics, changes parameters, uses query truth, applies class quotas, or globally reorders predictions. Every registered class receives a score. All40rows must finish writing predictions before the separate scorer opens query truth. Scores include accuracy, macro accuracy, old/new accuracy and their harmonic mean. Scoring results must never flow back into configuration or restart decisions.

Validation: Phase2 builder3tests passed; predictor/scorer9tests passed; independent P0/P1 review passed for source, data builder/config and Phase2 prediction candidate. Native DA/registration methods beyond these routes remain separate future batches; this document does not claim that every comparison in the wider publication plan has run.
