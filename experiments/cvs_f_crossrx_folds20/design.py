from pathlib import Path
from experiments.cvs_f_crossrx_practical20 import design as first
ROOT=Path(__file__).resolve().parents[2]
PROJECT,SOURCE,FULL_FP32_POLICY,read,write=(getattr(first,k) for k in ('PROJECT','SOURCE','FULL_FP32_POLICY','read','write'))
RUN='20261016-phase1-FcrossRX-practical-source-confirm-s2026092701-m8-r01'
RELEASE='cvs_FcrossRX_practical_source_confirm_E20_20261016_r01'
OWNER='codex/root/FcrossRX-practical-source-confirm-20261016'
BASE=Path(PROJECT)/'runs'/RUN
SEED=first.SEED;STEPS=222;UPDATES=4440;MAX_WORKERS=4
ARMS=first.ARMS;SCENES=first.SCENES;TRAIN_SCENES=first.TRAIN_SCENES;CHANNEL_SEEDS=dict(first.CHANNEL_SEEDS);SUPCON=dict(first.SUPCON)
ALL_RX=(1,3,4,6,8);NEW_HELD=(3,4,6,8)
REFERENCE=Path(PROJECT)/'runs'/first.RUN
def parent_config():return first.parent_config()
def rows():return [dict(arm=a,held_RX=held,row_id=f'{a}-RX{held}-s{SEED}') for held in NEW_HELD for a in ARMS]
def config(row):
    if row not in rows():raise ValueError('Exactly eight remaining source-fold confirmation rows allowed')
    c=first.config(next(r for r in first.rows() if r['arm']==row['arm']));fit=[r for r in ALL_RX if r!=row['held_RX']]
    c.update(schema='FcrossRX_practical_source_confirmation_E20_v1',run_id=RUN,**row,launch_owner=OWNER,
        output_root=(BASE/row['row_id']/'source').as_posix(),method='FcrossRX_paired_practical_confirmation',
        fold=dict(c['fold'],held_RX=row['held_RX'],fit_RX=fit),
        baseline_reference='Fresh same-held-RX C0; RX1 frozen result reused only for aggregate, never weights',
        confirmation=dict(reference_run=first.RUN,reference_held_RX=1,new_held_RX=list(NEW_HELD),
            same_source_rule_every_fold=True,all_five_folds_must_pass=True,model_seed_count=1,new_Ag=False,
            failure='No stable cross-RX claim or target promotion if any source fold fails; no failing fold/seed replacement'),
        cost_claim='Eight new scratchE20/35520updates/18186240Lnetwork packets; RX1 reference40modelE/8880updates separately reused;0parentweights',
        budget_authorization='Active mixedIQ goal; RX1 source rule PASS supports separately preregistered same-recipe four remaining source folds; E20 per model,0target')
    return c
def validate(c):
    row=next((r for r in rows() if r['row_id']==c.get('row_id')),None)
    if row is None or c!=config(row):raise ValueError('Fixed source confirmation/budget/defaulthead/role recipe differs')
def make_args(c,device='cuda:0'):
    validate(c);a=first.make_args(first.config(next(r for r in first.rows() if r['arm']==c['arm'])),device)
    a.run_id=RUN;a.candidate_id=c['row_id'];a.output_dir=c['output_root'];a.source_fold_held_rx=c['held_RX'];a.source_fold_training_receivers=','.join(map(str,c['fold']['fit_RX']))
    if (a.epochs,a.label_epochs,a.pseudo_epochs,a.label_smoothing)!=(20,13,7,.01) or a.baseline_ckpt or a.teacher_ckpt or a.use_concat_sat_channel_aug:raise ValueError('Exact scratchE20/oldU recipe required')
    return a
def reference_results():
    done=read(REFERENCE/'completion.json');q=read(REFERENCE/'source_decision.json')
    if done['status']!='ARTIFACTS_COMPLETE' or done['target_access'] or q['status']!='PASS':raise ValueError('Verified complete RX1 source PASS required; never reload reference weights')
    result={a:read(REFERENCE/f'{a}-RX1-s{SEED}'/'source/result.json') for a in ARMS}
    if any(r['held_RX']!=1 or r['fit_RX']!=[3,4,6,8] or r['optimizer_updates']!=4440 or r['target_access'] or r['U_truth_read'] or r['head_fit_updates'] for r in result.values()):raise ValueError('Reference source/result provenance differs')
    if first.qualify(result)!=q:raise ValueError('RX1 saved source decision differs')
    return result
def qualify(results,reference):
    if set(results)!=set(NEW_HELD):raise ValueError('All four held-source pairs required')
    per={1:first.qualify(reference)}
    for held in NEW_HELD:
        rr=results[held]
        if any(r['held_RX']!=held or r['fit_RX']!=[v for v in ALL_RX if v!=held] for r in rr.values()):raise ValueError('Result assigned to incorrect held source receiver')
        per[held]=first.qualify(rr)
    fields=('urban_gain_pp','low_gain_pp','urban_macroF1_gain_pp','clean_drop_pp','seen_drop_pp')
    return dict(status='PASS' if all(v['status']=='PASS' for v in per.values()) else 'FAIL',per_source_RX=per,
        five_fold_mean={k:sum(v[k] for v in per.values())/5 for k in fields},worst_clean_drop_pp=max(v['clean_drop_pp'] for v in per.values()),
        reused_reference_RX=1,new_source_folds=4,model_seed_count=1,source_only=True,target_access=False,automatic_extension=False,goal_completed=False)
