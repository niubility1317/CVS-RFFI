"""Separate cosine-compressed scratch CE baselines, matched to the original E200."""
from copy import deepcopy
from pathlib import Path
from experiments.cvs_feature_disentangle import design as original
from experiments.cvs_phase1_stack.design import args_from_validated_config

ROOT=Path(__file__).resolve().parents[2]
PROJECT=original.PROJECT;SOURCE=original.SOURCE;CAPSULE=original.CAPSULE;TRUTH=original.TRUTH
CLASSES=original.CLASSES;FULL_FP32_POLICY=original.FULL_FP32_POLICY;SEEDS=original.SEEDS
read,write=original.read,original.write
RUN='20261011-phase1-ce-epoch22-manysig-m3-r01'
RELEASE='cvs_ce_epoch22_20261011_r01'
BASE=Path(PROJECT)/'runs'/RUN
OWNER='codex/root/ce-epoch22-20261011'
BUDGETS=(22,);ARMS=tuple('E'+str(e) for e in BUDGETS)
PAIRS=[(a,'P0') for a in ARMS]


def rows():
    return [dict(row_id=f'E{e}-s{s}',epochs=e,model_seed=s) for s in SEEDS for e in BUDGETS]


def config(row):
    if row not in rows():raise ValueError('Unregistered epoch/seed row')
    seed=row['model_seed'];epoch=row['epochs']
    baseline_row=dict(stage='feature_disentangle',arm='native',model_seed=seed,row_id=f'native-s{seed}')
    c=deepcopy(original.config(baseline_row))
    c.update(schema='ce_epoch_compression_v1',stage='ce_epoch_compression',run_id=RUN,**row,
        method='cvs_ce_epoch22',budget_arm=f'E{epoch}',updates=epoch*222,
        output_root=(BASE/row['row_id']/'source').as_posix(),model_initialization='scratch',checkpoint_sources=[],
        checkpoint_selection='final_only',test_views=['clean'],launch_owner=OWNER)
    c['risk_recipe'].update(epochs=epoch,source_selection='fixed final epoch; all three rows tested')
    c['concurrency']=dict(max_processes_per_gpu=3,includes_existing=True,gpu_ids=list(range(8)))
    reference=Path(PROJECT)/'runs/20261010-phase1-feature-disentangle-manysig-m48-r02'/f'native-s{seed}'
    c['reference_E200']=dict(checkpoint=(reference/'source/final_ssdg.pth').as_posix(),prediction=(reference/'prediction/predictions.npz').as_posix(),role='comparison only; never loaded for initialization')
    c['source_selection']=dict(rule='fixed final epoch; every predeclared row tested; no source V epoch selection',
        future_compression='USER EXPLICIT EXCEPTION: clean three-seed mean no more than 0.003 below fixed original E200 permits shorter epoch exploration',
        benchmark_role='exposed development benchmark for epoch budget only; not independent blind test',
        target_samples_or_labels_in_training=False,noninferiority_accuracy_margin=.003)
    return c


def validate(c):
    row=next((r for r in rows() if r['row_id']==c.get('row_id')),None)
    if row is None or c!=config(row):raise ValueError('Epoch compression configuration differs')
    if c['arm']!='native' or c['features'] or c['arm_plan']['paths'] or c['feature_plan']['relation']!='none' or c['feature_plan']['style'] or c['feature_plan']['fishr']!='none':raise ValueError('Pure baseline mechanisms differ')
    return c


def make_args(c,device='cuda:0'):
    validate(c);a=args_from_validated_config(c,device)
    a.epochs=c['epochs'];a.label_epochs=c['epochs'];a.pseudo_epochs=0
    a.label_smoothing=0.;a.use_unlabeled=False;a.use_ema_teacher=False;a.lambda_u=0.;a.lambda_ent=0.
    a.pseudo_domain_gate=False;a.pseudo_temporal_gate=False;a.pseudo_strong_agreement=False
    a.phase1_lr_schedule='cosine';a.phase1_lr_min=1e-6
    return a
