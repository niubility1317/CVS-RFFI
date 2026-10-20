from experiments.pamf_compact_identity_probe import design as base

RUN='20261020-diagnostic-restoration-physics-compact-analytic-TX-s2026092701-m2-r42'
PARENT='20261020-phase1-restoration-IQ-physics-compact-s2026092701-m2-r41'
PARENT_COMMIT='be54678266c95e28bbc18f5a71387c8becf463fa'
REFERENCE,REFERENCE_COMMIT=base.REFERENCE,base.REFERENCE_COMMIT
ACTION,ACTION_COMMIT=base.ACTION,base.ACTION_COMMIT
VIEWS,GROUPS,NEW_ROWS,PARAMETERS=base.VIEWS,base.GROUPS,base.NEW_ROWS,base.PARAMETERS
CANDIDATES,READOUTS,COMPARISONS=base.CANDIDATES,base.READOUTS,base.COMPARISONS


def configuration(commit):
    value=base.configuration(commit)
    value.update(schema='physics_compact_F_independent_analytic_identity_v1',run=RUN,parent=PARENT,parent_commit=PARENT_COMMIT,
        primary_question='Do both physically anchored F-free compact source IQs retain cross-RX-and-day sameTX association in the original fixed R34 coordinates?',
        wait_for='Both R41 fixedE20/4940 source_complete only;no source/target scores or target completion inspected;share R41score.lock CPU capacity',
        claim_scope='Fixed unchanged R34/R39 analytic association andpaired error,not a fitted classifier,formalaccuracy orTX-exclusive/per-detail proof',
        parent_training_or_decisions_changed=False,source_method_changed=False,new_calibration_samples=0)
    return value
