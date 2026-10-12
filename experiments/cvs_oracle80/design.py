"""User-authorized fully supervised 80/20 reference, separate from weak labels."""
from copy import deepcopy
from pathlib import Path
from experiments.cvs_ce_epoch2040 import design as baseline

ROOT=Path(__file__).resolve().parents[2]
PROJECT=baseline.PROJECT
RUN='20261012-phase1-oracle80-e30-manysig-s2026092701-r01'
RELEASE='cvs_oracle80_e30_20261012_r01'
BASE=Path(PROJECT)/'runs'/RUN
SOURCE=(Path(PROJECT)/'releases'/RELEASE/'source_contract.json').as_posix()
CAPSULE=baseline.CAPSULE;TRUTH=baseline.TRUTH
CLASSES=baseline.CLASSES;FULL_FP32_POLICY=baseline.FULL_FP32_POLICY
SEEDS=(2026092701,);OWNER='codex/root/oracle80-e30-20261012'
read,write=baseline.read,baseline.write
SCENES=('practical_high','practical_mid','practical_low_suburban','practical_high_urban','practical_mid_urban','practical_low_urban')
VIEWS=('clean',*SCENES)
VIEWS_ROOT=Path(PROJECT)/'runs/20261002-phase1-cvs-residual-sixscene-manysig-m8-r01/received_views'
COUNT=168000

def rows():return [dict(row_id='E30-s2026092701',epochs=30,model_seed=2026092701)]

def config(row):
    if row not in rows():raise ValueError('Unregistered oracle row')
    c=deepcopy(baseline.config(dict(row_id='E30-s2026092701',epochs=30,model_seed=2026092701)))
    c.pop('reference_E200',None)
    c.update(schema='oracle80_e30_v1',stage='oracle80_e30',run_id=RUN,**row,
        method='cvs_oracle80',output_root=(BASE/row['row_id']/'source').as_posix(),source_contract=SOURCE,
        steps_per_epoch=563,updates=16890,launch_owner=OWNER,test_views=list(VIEWS),
        oracle_data=dict(source_rx=[1,3,4,6,8],source_days=[1,2,3],target_rx=[0,2,5,7,9,10,11],
            target_days=[0,1,2,3],L_s=72000,U_s=0,V=18000,ratios=[.8,0.,.2],rho_label=1.,
            split='per TX/RX/day/eq sorted sig_i; first800 train,last200 V',equalized=1),
        source_selection=dict(rule='fixed final E30; source V health only; no target-feedback selection'),
        epoch_definition='one shuffled full pass of all72000 train packets; batch128,last64;563 updates',
        concurrency=dict(max_processes_per_gpu=3,includes_existing=True,gpu_ids=list(range(8))))
    c['risk_recipe'].update(epochs=30,steps_per_epoch=563,source_selection='fixed final E30; full-label supervised reference')
    c['feature_recipe'].update(source_rx=[1,3,4,6,8],source_days=[1,2,3])
    return c

def validate(c):
    if c!=config(rows()[0]):raise ValueError('Oracle configuration changed')
    return c

def make_args(c,device='cuda:0'):
    from experiments.cvs_phase1_stack.design import args_from_validated_config
    validate(c);a=args_from_validated_config(c,device)
    for k,v in dict(epochs=30,label_epochs=30,pseudo_epochs=0,label_smoothing=0.,use_unlabeled=False,
        use_ema_teacher=False,lambda_u=0.,lambda_ent=0.,pseudo_domain_gate=False,pseudo_temporal_gate=False,
        pseudo_strong_agreement=False,phase1_lr_schedule='cosine',phase1_lr_min=1e-6,
        labeled_ratio=.8,unlabeled_ratio=0.,source_val_ratio=.2,u_unlabeled_shuffle=False,
        phase1_source_role_protocol='legacy_l_u_v').items():setattr(a,k,v)
    return a
