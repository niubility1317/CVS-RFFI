import copy
from experiments.pamf_factor_formal_identity import design as old
from experiments.pamf_r46_response_distill import design as r26
from experiments.pamf_r46_prior_collapse import design as r28
from experiments.pamf_r46_moment_fidelity import design as r30
from experiments.pamf_r46_gradient_coordination import design as r32

RUN='20261021-diagnostic-restoration-response-formal-analytic-TX-s2026092701-m8-r33'
PARENT,PARENT_COMMIT=r32.RUN,'8f2e8a3fb413d0d116ec6ee3bb7f9de871d3ba83'
PARENTS={r26.RUN:dict(commit='781e56d9d2bd401d56ad7b758ed4f4812c04f1e4',rows=list(r26.ARMS)),
    r28.RUN:dict(commit='7ca5af4f6f5d9f862e07e13b9da15902379044e4',rows=list(r28.ARMS)),
    r30.RUN:dict(commit='0b2c778d566ef83f159f236328cc704897a24507',rows=list(r30.ARMS)),
    r32.RUN:dict(commit=PARENT_COMMIT,rows=list(r32.ARMS))}
ROWS={}
for tag,module in [('r26',r26),('r28',r28),('r30',r30),('r32',r32)]:
    for arm in module.ARMS:
        ROWS[tag+'_'+arm]=dict(run=module.RUN,row=arm,parameters=module.PARAMETERS[arm],updates=module.UPDATES,
            kind='frozen_prior_collapse' if tag=='r28' else 'trained')
NEW_ROWS=tuple(ROWS)
parent=old.parent
BASELINE,BASELINE_COMMIT,BASELINE_ROW=old.BASELINE,old.BASELINE_COMMIT,old.BASELINE_ROW
REFERENCE,REFERENCE_COMMIT=old.REFERENCE,old.REFERENCE_COMMIT
FORMAL,TRUTH_RUN,VIEWS,GROUPS,COUNT=old.FORMAL,old.TRUTH_RUN,old.VIEWS,old.GROUPS,old.COUNT
CACHED_RUN,CACHED_COMMIT=old.RUN,'8a2899f5998bd50d1595391f8c7fad9667b0d1fa'
CACHED_FEATURE_READOUTS=list(old.READOUTS)
CACHED_READOUTS=dict(raw='raw',r46_350521='r46_350521',r22_complex625='r22_complex625',r22_complex625_identity='r22_complex625_identity')
CANDIDATES=('r46_350521','r22_complex625','r22_complex625_identity',*NEW_ROWS)
READOUTS=('raw',*CANDIDATES)
COMPARISONS=tuple([(a,'r46_350521') for a in NEW_ROWS]+[
    ('r26_odd_response','r26_pair_aug_control'),
    ('r28_R22_complex625_priorcollapsed','r22_complex625'),
    ('r28_R22_complex625_identity_priorcollapsed','r22_complex625_identity'),
    ('r30_moment_teacher','r30_moment_clean'),
    ('r32_moment_teacher_symproj','r32_moment_teacher_grad_sum'),
    ('r32_moment_teacher_grad_sum','r30_moment_teacher'),
    ('r32_moment_teacher_symproj','r30_moment_teacher'),
    ('r26_pair_aug_control','r22_complex625_identity'),
    ('r26_odd_response','r22_complex625_identity'),
    ('r30_moment_clean','r22_complex625_identity'),
    ('r30_moment_teacher','r22_complex625_identity')])
SCORING_RUN=old.SCORING_RUN
CACHED_CELLS=len(CACHED_READOUTS)*len(VIEWS)*len(GROUPS)
NEW_CELLS=len(NEW_ROWS)*len(VIEWS)*len(GROUPS)
TOTAL_CELLS=CACHED_CELLS+NEW_CELLS
EXPECTED_RECORDS=len(CANDIDATES)*len(VIEWS)*len(GROUPS)*(1+6+7+4)
EXPECTED_CONTROLS=len(CANDIDATES)*len(VIEWS)*len(GROUPS)*3
EXPECTED_COMPARISONS=len(COMPARISONS)*len(VIEWS)*len(GROUPS)


def configuration(commit):
    cfg=copy.deepcopy(old.configuration(commit))
    cfg.update(schema='response_formal_F_independent_identity_v1',run=RUN,parent=PARENT,parent_commit=PARENT_COMMIT,parents=PARENTS,rows=ROWS,
        candidates=list(CANDIDATES),readouts=list(READOUTS),cached_run=CACHED_RUN,cached_commit=CACHED_COMMIT,cached_feature_cells=CACHED_CELLS,
        new_feature_cells=NEW_CELLS,total_feature_cells=TOTAL_CELLS,comparisons=[list(v) for v in COMPARISONS],
        expected_records=EXPECTED_RECORDS,expected_controls=EXPECTED_CONTROLS,expected_comparisons=EXPECTED_COMPARISONS,
        label_use='Evaluationstatistics only after all336featurecells fixed and all8newrows full7formal scores complete;not training labels',
        wait_rule='All R26/R28/R30/R32 complete8formal rows before savedtargetIQ/feature access;never gates/delays parent jobs',
        inclusion='All6 trained and2frozenreparameterized versions;R28zero new updates/sourceR22ancestry explicit;no score selection or omission of negatives',
        coordinates='Unchanged R24/R34/R50 mirror64/spectral4/reference24/combined92;reuse originalLU8192 moments/std.05;0new fit',
        cached_scope='112 existingR24 cells raw/original350521/nativeR22both;224newcells;0extra neuralprediction',
        parent_training_or_decisions_changed=False,goal_complete=False)
    return copy.deepcopy(cfg)
