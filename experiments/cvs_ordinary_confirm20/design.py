"""Confirmation fixed by the successful RX1 source gate; no target inputs."""
from pathlib import Path
from experiments.cvs_ordinary_fold20 import design as old

ROOT=Path(__file__).resolve().parents[2]
PROJECT,SOURCE,FULL_FP32_POLICY,read,write=(getattr(old,k) for k in ('PROJECT','SOURCE','FULL_FP32_POLICY','read','write'))
RUN='20261016-phase1-ordinary-augmentation-sourceRX-confirm-e20-s2026092701-m8-r01'
RELEASE='cvs_ordinary_augmentation_sourceRX_confirm_e20_20261016_r01'
OWNER='codex/root/ordinary-sourceRX-confirm-E20-20261016'
BASE=Path(PROJECT)/'runs'/RUN
SEED=old.SEED;UPDATES=4440;STEPS=222
RX=(1,3,4,6,8);HELD=(3,4,6,8);ARMS=('N0','N1')
FIRST_ROOT=old.BASE.as_posix()


def rows():return [dict(arm=a,held_RX=r,row_id=f'{a}-RX{r}-s{SEED}') for r in HELD for a in ARMS]
def parent_config():return old.parent_config()


def config(row):
    if row not in rows():raise ValueError('Only four remaining source RX, two fixed arms permitted')
    base=old.config(next(r for r in old.rows() if r['arm']==row['arm']))
    fold=dict(base['fold'],held_RX=row['held_RX'],fit_RX=[r for r in RX if r!=row['held_RX']])
    return dict(base,schema='ordinary_augmentation_remaining_source_RX_v1',run_id=RUN,**row,
        stage='source_RX_confirmation',method='cvs_ordinary_source_RX_confirmation',launch_owner=OWNER,
        output_root=(BASE/row['row_id']/'source').as_posix(),fold=fold,
        baseline_reference='Fresh N0 of this same source fold; RX1 predictions reused only for five-fold aggregate, never weights',
        training_recipe=f'Current SIG_TIME RAW_HZ/LS.01/NoExplicitCFO/joint160/defaultcos/RDrop1, scratch20=13L+7U, heldsourceRX{row["held_RX"]}, ordinaryL '+row['arm'],
        source_falsification=dict(aggregate='Fixed five source RX folds, including completed RX1; no individual fold selection',
            mean_accuracy_gain_pp_min=1.,positive_folds_min=4,worst_accuracy_gain_pp_min=-1.,
            mean_macroF1_gain_min=0.,seen_V_mean_drop_pp_each_max=.5,
            finite_complete_updates=True,full_TX_RX_day_coverage=True,matching_real_L_U_sequence=True,
            on_fail='End ordinary-as-Ag hypothesis, preserve every fold and class loss; no heldRX/seed/strength rescue',
            on_pass='Source evidence only; separate full-source joint design still required, no target/seed auto-extension'),
        budget_authorization='RX1 preregistered source gate passed; fixed remaining4 source folds,8 independently scratchE20 models, one model seed; no E50/E200 weights or budget extension per model',
        cost_claim='New8x20=160model-epochs/35520updates; completedRX1x2=40model-epochs reused as evidence. Cumulative200model-epochs are10 separateE20 models, not oneE200 trajectory. No inherited weights.')


def validate(c):
    row=next((r for r in rows() if r['row_id']==c.get('row_id')),None)
    if row is None or c!=config(row):raise ValueError('Fixed remaining-source-RX configuration differs')


def make_args(c,device='cuda:0'):
    validate(c);base=old.config(next(r for r in old.rows() if r['arm']==c['arm']));a=old.make_args(base,device)
    a.run_id=RUN;a.candidate_id=c['row_id'];a.output_dir=c['output_root']
    a.source_fold_held_rx=c['held_RX'];a.source_fold_training_receivers=','.join(map(str,c['fold']['fit_RX']))
    return a


def prerequisite():
    root=Path(FIRST_ROOT);complete=read(root/'complete.json');decision=read(root/'source_decision.json')
    if complete['status']!='ARTIFACTS_COMPLETE' or not complete['source_only'] or complete['target_access'] or decision['status']!='PASS' or decision['formal_target_qualified']:
        raise ValueError('Verified source-only RX1 PASS required before confirmation')
    result={}
    for arm in ARMS:
        source=root/f'{arm}-RX1-s{SEED}'/'source';value=read(source/'result.json');done=read(source/'completion.json')
        init=read(source/'initialization.json');contract=read(source/'source_contract.json')
        if (done['epoch']!=20 or done['optimizer_steps']!=4440 or done['initialization']!='scratch' or done['checkpoint_sources']
            or done['ancestors'] or done['target_access'] or done['head_fit_updates'] or not init['scratch_only'] or init['target_contact']
            or value['held_V']['count']!=5400 or value['target_access'] or value['U_truth_read'] or value['head_fit_updates']
            or contract['held_RX']!=1 or {k:len(v) for k,v in contract['role_ids'].items()}!={'L_s':5040,'U_s':45360,'V':21600}):
            raise ValueError('Completed RX1 reuse must retain exact source/default-head/E20 contract')
        result[arm]=value
    if old.qualify(result)['status']!='PASS':raise ValueError('RX1 source qualification arithmetic differs')
    return dict(status='VERIFIED',root=FIRST_ROOT,results=result,weights_loaded=False,target_access=False,
        reused_model_epochs=40,reused_updates=8880,scope='Read-only source result reuse, never initialization/teacher state')


def qualify(folds):
    if set(folds)!=set(RX) or any(set(v)!=set(ARMS) for v in folds.values()):raise ValueError('All five fixed source RX pairs required')
    records=[]
    for rx in RX:
        a,b=folds[rx]['N1'],folds[rx]['N0']
        if any(v['held_V']['count']!=5400 or v['target_access'] or v['U_truth_read'] or v['head_fit_updates'] for v in (a,b)):
            raise ValueError('Every source pair must be complete and legal')
        records.append(dict(held_RX=rx,accuracy_gain_pp=100*(a['held_V']['accuracy']-b['held_V']['accuracy']),
            macroF1_gain=a['held_V']['macro_f1']-b['held_V']['macro_f1'],
            seen_V_mean_drop_pp=100*(b['seen_V']['source_val_accuracy']-a['seen_V']['source_val_accuracy']),
            healthy=bool(a['complete_health'] and b['complete_health'])))
    mean=sum(v['accuracy_gain_pp'] for v in records)/5;f1=sum(v['macroF1_gain'] for v in records)/5
    checks=dict(mean_accuracy=mean>=1.-1e-9,positive_folds=sum(v['accuracy_gain_pp']>0 for v in records)>=4,
        worst_fold=min(v['accuracy_gain_pp'] for v in records)>=-1.-1e-9,mean_macroF1=f1>=-1e-12,
        seen_health=all(v['seen_V_mean_drop_pp']<=.5+1e-9 for v in records),complete_health=all(v['healthy'] for v in records))
    return dict(status='PASS' if all(checks.values()) else 'FAIL',checks=checks,folds=records,
        mean_accuracy_gain_pp=mean,mean_macroF1_gain=f1,model_seed_count=1,source_fold_count=5,
        source_only=True,target_access=False,formal_target_qualified=False,automatic_extension=False,goal_completed=False)
