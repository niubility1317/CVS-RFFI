from pathlib import Path
from copy import deepcopy
from experiments.cvs_report_ssl import design as original
ROOT=Path(__file__).resolve().parents[2]
PROJECT,SOURCE,CAPSULE,TRUTH,CLASSES=original.PROJECT,original.SOURCE,original.CAPSULE,original.TRUTH,original.CLASSES
SEEDS=original.SEEDS;FULL_FP32_POLICY=original.FULL_FP32_POLICY
read,write=original.read,original.write
RUN='20261011-phase1-stage2-teacher-factorial-manysig-m24-r01'
RELEASE='cvs_stage2_teacher_factorial_20261011_r01'
BASE=Path(PROJECT)/'runs'/RUN
OWNER='codex/root/stage2-teacher-factorial-20261011'
ARMS=tuple(ref+'_'+mode for ref in ('O','R') for mode in ('CE','EMA','DIRECT','DIRECT_HI'))
PRIMARY='O_DIRECT_HI'
PAIRS=[(ref+'_'+mode,ref+'_CE') for ref in ('O','R') for mode in ('EMA','DIRECT','DIRECT_HI')]+[(ref+'_DIRECT',ref+'_EMA') for ref in ('O','R')]+[(ref+'_DIRECT_HI',ref+'_DIRECT') for ref in ('O','R')]+[('R_'+mode,'O_'+mode) for mode in ('CE','EMA','DIRECT','DIRECT_HI')]+[(a,'P0') for a in ARMS]
learning_rate=original.learning_rate
frequency_learning_rate=original.frequency_learning_rate

def rows():return [dict(arm=a,model_seed=s,row_id=f'{a}-s{s}') for s in SEEDS for a in ARMS]

def config(row):
    if row not in rows():raise ValueError('Unregistered teacher-factorial row')
    arm=row['arm'];ref,mode=arm.split('_',1);base='C0' if mode=='CE' else 'C1'
    c=deepcopy(original.config(dict(arm=base,model_seed=row['model_seed'],row_id=f"{base}-s{row['model_seed']}")))
    c.update(schema='stage2_teacher_factorial_v1',run_id=RUN,**row,stage='phase1_stage2_teacher_factorial',method='cvs_stage2_teacher_factorial',method_name=arm,output_root=(BASE/row['row_id']/'source').as_posix(),launch_owner=OWNER,teacher=mode=='EMA',unlabeled=mode!='CE',epochs=40,updates=8880)
    c['semi'].update(target_provider='EMA' if mode=='EMA' else 'current_student' if mode!='CE' else 'none',ema_decay=.99 if mode=='EMA' else None,teacher_init='after update888: cloned EMA or live stop-gradient student according to target_provider',conditional_refresh_epochs=[4,12,20,28,36])
    if ref=='O':c['frontend']=dict(version='original_reference_response_v1',reference_dim=29,zero_slot=None,state_dim=40,explicit_frequency_estimator='inherited original baseline reference response',raw_IQ_retained=True,whole_model_frequency_invariant=False)
    c['reference_mode']='original' if ref=='O' else 'report_revised'
    if mode=='DIRECT_HI':c['optimizer'].update(encoder_peak_lr=3e-4,fusion_head_peak_lr=9e-4)
    c['evaluation']=dict(primary_arm=PRIMARY,goal_three_seed_mean=.84,all_predeclared_arms_reported=True,truth_last=True,feedback_to_training=False,fixed_control_run=None,fixed_control_arms=[],control_role='full 2x4 fixed factorial, matched original E200 parents; no epoch-selected parent')
    c.pop('C5_selection',None)
    c['mechanism_scope']='No new auxiliary losses in this causal control batch; no EMA branch creates no averaged copy. Inherited old reference only in O rows, original gates unchanged throughout.'
    return c

def validate(c):
    row=next((r for r in rows() if r['row_id']==c.get('row_id')),None)
    if row is None or c!=config(row):raise ValueError('Teacher-factorial config differs')
    return c
