from copy import deepcopy
from pathlib import Path
from experiments.cvs_ce_epoch2040 import design as baseline
from experiments.cvs_phase1_stack.design import args_from_validated_config

ROOT=Path(__file__).resolve().parents[2]
PROJECT=baseline.PROJECT
RUN='20261012-phase1-e20-historical-adv3b02-manysig-s392002-r02'
RELEASE='cvs_e20_historical_adv3b02_20261012_r02'
BASE=Path(PROJECT)/'runs'/RUN
SOURCE=(Path(PROJECT)/'releases'/RELEASE/'source_contract.json').as_posix()
CAPSULE=(BASE/'data/capsule').as_posix();TRUTH=(BASE/'data/truth.npz').as_posix()
CLASSES=baseline.CLASSES;FULL_FP32_POLICY=baseline.FULL_FP32_POLICY
SEEDS=(392002,);OWNER='codex/root/historical-adv3b02-e20-20261012'
read,write=baseline.read,baseline.write
SPLITS={'unseen_day_seen_rx':84000,'seen_day_unseen_rx':60000,'unseen_day_unseen_rx':60000}
COUNT=sum(SPLITS.values())

def rows():return [dict(row_id='E20-s392002',epochs=20,model_seed=392002)]

def config(row):
    if row not in rows():raise ValueError('Unregistered historical row')
    c=deepcopy(baseline.config(dict(row_id='E20-s2026092701',epochs=20,model_seed=2026092701)))
    c.pop('reference_E200',None)
    c.update(schema='historical_adv3b02_e20_v1',stage='historical_adv3b02_e20',run_id=RUN,**row,
        method='cvs_adv3b02_e20',output_root=(BASE/row['row_id']/'source').as_posix(),source_contract=SOURCE,
        historical_data=dict(source_rx=list(range(7)),source_days=[0,1],target_rx=list(range(7,12)),target_days=[2,3],
            L_s=8400,U_s=58800,V=16800,ratios=[.1,.7,.2],rho_label=.125,equalized=1,
            split='per TX/RX/day/eq sorted sig_i; first100 L,next700 U,last200 V'),
        launch_owner=OWNER,test_views=['clean'],test_splits=SPLITS,
        source_selection=dict(rule='fixed final E20; V health only; no target-feedback selection'),
        concurrency=dict(max_processes_per_gpu=3,includes_existing=True,gpu_ids=list(range(8))))
    c['feature_recipe'].update(source_rx=list(range(7)),source_days=[0,1])
    c['risk_recipe']['source_selection']='fixed final E20; historical seed392002'
    return c

def validate(c):
    if c!=config(rows()[0]):raise ValueError('Historical E20 configuration changed')
    return c

def make_args(c,device='cuda:0'):
    validate(c);a=args_from_validated_config(c,device)
    for k,v in dict(epochs=20,label_epochs=20,pseudo_epochs=0,label_smoothing=0.,use_unlabeled=False,
        use_ema_teacher=False,lambda_u=0.,lambda_ent=0.,pseudo_domain_gate=False,pseudo_temporal_gate=False,
        pseudo_strong_agreement=False,phase1_lr_schedule='cosine',phase1_lr_min=1e-6,
        wisig_train_rxs='0,1,2,3,4,5,6',wisig_test_rxs='7,8,9,10,11',wisig_train_days='0,1',
        wisig_test_days='2,3',allow_source_target_day_overlap=False,split_mode='tx_rx_day_1_7_2',
        labeled_ratio=.10,unlabeled_ratio=.70,source_val_ratio=.20,
        phase1_source_role_protocol='legacy_l_u_v').items():setattr(a,k,v)
    return a
