from experiments.pamf_analytic_identity_probe import design as base

RUN='20261020-diagnostic-restoration-IQ-compact-analytic-TX-s2026092701-m2-r39'
PARENT='20261020-phase1-restoration-r17-IQ-joint-compact-s2026092701-m2-r38'
PARENT_COMMIT='e0868f0a0c3cf056a1753a5b4e5ce4f0bff84381'
REFERENCE=base.RUN
REFERENCE_COMMIT='60f0d924aa0a43f4213c683df17d7ae9e2ea54ee'
ACTION=base.ACTION
ACTION_COMMIT=base.ACTION_COMMIT
VIEWS=base.VIEWS
GROUPS=base.GROUPS
NEW_ROWS=('dense_three_quarter','dense_half')
PARAMETERS=dict(zip(NEW_ROWS,(1709713,794529)))
CANDIDATES=('r17_reference',*NEW_ROWS)
READOUTS=('raw',*CANDIDATES)
COMPARISONS=(('dense_half','dense_three_quarter'),('dense_three_quarter','r17_reference'),('dense_half','r17_reference'))


def configuration(commit):
    value=base.configuration(commit)
    value.update(schema='compact_F_independent_analytic_identity_v1',run=RUN,parent=PARENT,parent_commit=PARENT_COMMIT,
        candidate=None,candidates=list(CANDIDATES),readouts=list(READOUTS),comparisons=[list(p) for p in COMPARISONS],
        reference_run=REFERENCE,reference_commit=REFERENCE_COMMIT,new_calibration_samples=0,
        calibration='Reuse exact fixed R34 first8192 LU moments,groups,coordinates andraw/R17features;no refit or newcalibrationreads',
        wait_for='Both compact source rows E20/4940 andsource_complete;no target metrics or targetcompletion required;share parent score.lock CPU capacity',
        primary_question='Do both F-free compact restorers retain cross-RX-and-day sourceTX association vs originalR17 in fixed analytic coordinates?',
        new_source_thresholds=False,parent_training_or_decisions_changed=False,
        claim_scope='Same fixed R34 descriptive source probe;allgroups retained,including spectralclean negative;no allTX/perdetail proof',
        stop_rule='Wrong source parent/roles/panel,missing or nonfinite input/features,outputcollision:preserve outputs,noautomaticretry;healthy compact/peers unchanged')
    return value
