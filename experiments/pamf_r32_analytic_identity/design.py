from experiments.pamf_analytic_identity_probe import design as base

RUN='20261020-diagnostic-restoration-R32-analytic-identity-s2026092701-r35'
PARENT='20261020-phase1-restoration-observable-response-selfsup-s2026092701-m2-r32'
PARENT_COMMIT='1857f2c26eabbf68838bad1074e4fdc9b02e267f'
REFERENCE=base.RUN
REFERENCE_COMMIT='60f0d924aa0a43f4213c683df17d7ae9e2ea54ee'
ACTION=base.ACTION
ACTION_COMMIT=base.ACTION_COMMIT
VIEWS=base.VIEWS
GROUPS=base.GROUPS
NEW_ROWS=('pair_aug_control','observable_response')
CANDIDATES=('r17_reference',*NEW_ROWS)
READOUTS=('raw',*CANDIDATES)
COMPARISONS=(('observable_response','pair_aug_control'),('pair_aug_control','r17_reference'),('observable_response','r17_reference'))


def configuration(commit):
    value=base.configuration(commit)
    value.update(schema='R32_F_independent_analytic_identity_v1',run=RUN,parent=PARENT,parent_commit=PARENT_COMMIT,
        candidate=None,candidates=list(CANDIDATES),readouts=list(READOUTS),comparisons=[list(p) for p in COMPARISONS],
        reference_run=REFERENCE,reference_commit=REFERENCE_COMMIT,new_calibration_samples=0,
        calibration='Reuse exact fixed R34 first8192 LU moments,groups,coordinates andraw/R17features;no refit or newcalibrationreads',
        wait_for='Both R32 source rows E20/4940 andsource complete;no requirement for parent target decision/completion',
        primary_question='Does matched response constraint improve actualsourceTX analytic association as well as syntheticdirectionresponse?',
        new_source_thresholds=False,parent_training_or_decisions_changed=False,
        claim_scope='Same fixed R34 descriptive source probe;allgroups retained,including spectralclean negative;no allTX/perdetail proof',
        stop_rule='Wrong source parent/roles/panel,missing or nonfinite input/features,outputcollision:preserve outputs,noautomaticretry;healthy R32/peers unchanged')
    return value
