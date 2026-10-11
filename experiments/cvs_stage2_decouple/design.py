"""Immutable twelve-row plan; target scores never enter configuration."""
from pathlib import Path
from copy import deepcopy
from experiments.cvs_phase1_overlay.contract import PROJECT, SOURCE, CAPSULE, TRUTH, CLASSES, FULL_FP32_POLICY, read, write
from experiments.cvs_phase1_stack.design import args_from_validated_config

ROOT=Path(__file__).resolve().parents[2]
RUN='20261011-phase1-stage2-decouple-manysig-m12-r01'
RELEASE='cvs_stage2_decouple_20261011_r02'
BASE=Path(PROJECT)/'runs'/RUN
OWNER='codex/root/stage2-decouple-20261011'
SEEDS=(2026092701,2026092702,2026092703)
ARMS=('C0','C1','C2','C3')
PARENT_RUN='20261010-phase1-feature-disentangle-manysig-m48-r02'
PARENT_ORIGIN='20261010-phase1-feature-disentangle-manysig-m48-r01'
PARENT_COMMIT='d4723b1aa227fce6368fd25680bb74b1190557d7'

def rows():
    return [dict(stage='stage2_decouple',arm=a,model_seed=s,row_id=f'{a}-s{s}') for s in SEEDS for a in ARMS]

def config(row):
    p=Path(PROJECT)/'runs'/PARENT_RUN/f"native-s{row['model_seed']}"
    return dict(schema='stage2_decouple_v1',run_id=RUN,**row,method='cvs_stage2_decouple',variant='reference_response',features=[],
        dataset=PROJECT+'/Dataset_WigSig/ManySig.pkl',source_contract=SOURCE,
        output_root=(BASE/row['row_id']/'source').as_posix(),parent_checkpoint=(p/'source/final_ssdg.pth').as_posix(),
        parent_prediction=(p/'prediction/predictions.npz').as_posix(),
        preparation=(BASE/'prepared'/f"s{row['model_seed']}").as_posix(),
        model_initialization='strict_matched_pure_CE_parent_fresh_optimizer',checkpoint_sources=[(p/'source/final_ssdg.pth').as_posix()],
        epochs=40,steps_per_epoch=222,batch_size=128,updates=8880,label_smoothing=0.,
        optimizer=dict(name='AdamW',weight_decay=.0001,encoder_peak_lr=.00003,fusion_head_peak_lr=.0001,min_lr=.000001,warmup_steps=444),
        losses=dict(keep=.1,aug_ce=.25,drop=.1,receiver=.1,delta_keep=.5,delta_drop=.5,huber_delta=1.),
        schedule=dict(action_every=4,receiver_every=8,relation_batch=32,action_batch=32,action_order=['linear','linear','temporal','temporal']),
        enabled=dict(keep=row['arm']!='C0',actions=row['arm'] in ('C2','C3'),receiver=row['arm']=='C3'),
        numerical_policy=deepcopy(FULL_FP32_POLICY),target_access=False,test_views=['clean'],checkpoint_selection='final_only',
        sampling='shared seed-private source-L stream; 96 ordinary + 2x2x8 relation slots every8; no difficulty sampling',
        auxiliary=dict(action_passes=2,action_fit_batch=128,action_fit_lr=.0002,action_fit_wd=.0001,receiver_fit_steps=200,
                       receiver_fit_lr=.002,receiver_fit_wd=0.,fractional_delay_samples=.25),
        evaluation=dict(primary_arm='C3',goal_three_seed_mean=.84,truth_last=True,feedback_to_training=False),
        concurrency=dict(max_processes_per_gpu=4,includes_existing=True),launch_owner=OWNER)

def validate(c):
    r=next((r for r in rows() if r['row_id']==c.get('row_id')),None)
    if r is None or c!=config(r):raise ValueError('Unregistered or modified stage2 configuration')
    return c

def make_args(c,device='cuda:0'):
    validate(c)
    a=args_from_validated_config(c,device)
    a.epochs=a.label_epochs=40;a.pseudo_epochs=0;a.label_smoothing=0.
    a.use_unlabeled=False;a.use_ema_teacher=False
    a.from_scratch=False;a.a1_scratch_only=False;a.baseline_ckpt=c['parent_checkpoint']
    a.batch_size=128;a.num_workers=0
    a.phase1_lr_schedule='cosine';a.phase1_lr_min=1e-6
    return a
