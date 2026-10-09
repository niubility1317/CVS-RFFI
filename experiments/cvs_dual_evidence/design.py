"""Fixed four-arm architectural comparison; unchanged legacy pseudo recipe."""
from pathlib import Path
from experiments.cvs_phase1_repair import design as baseline
from experiments.cvs_phase1_stack.source import require_budget
from experiments.cvs_phase1_stack.design import PROJECT,SOURCE,CAPSULE,TRUTH,SEEDS,CLASSES,FULL_FP32_POLICY,read,write,args_from_validated_config

ROOT=Path(__file__).resolve().parents[2]
RUN='20261009-phase1-dual-evidence-manysig-m16-r01'
RELEASE='cvs_dual_evidence_20261009_r01'
BASE=Path(PROJECT)/'runs'/RUN
OWNER='codex/root/dual-evidence-20261009'
ARMS=('legacy_cosine','raw_dual','curvature_dual','curvature_interaction')
RECIPE=dict(lags=[1,2,4,8],evidence_channels=12,widths=[32,64,96,160],fusion_rank=64,
    curvature_log_scale=.25,interaction_bound=.5,initial_fusion_output='zero',
    new_objective_weights={},source_selection='fixed_E200_all_controls')


def rows():return [dict(stage='dual_evidence',arm=a,model_seed=s,row_id=a+'-s'+str(s)) for s in SEEDS for a in ARMS]


def config(row):
    c=baseline.config(dict(stage='repair',arm='pseudo_batch_cosine',model_seed=row['model_seed'],row_id='pseudo_batch_cosine-s'+str(row['model_seed'])))
    c.update(schema='dual_evidence_v1',run_id=RUN,**row,method='cvs_dual_evidence',
        output_root=(BASE/row['row_id']/'source').as_posix(),dual_recipe=RECIPE,
        baseline_recipe='cvs_phase1_repair/pseudo_batch_cosine',checkpoint_sources=[])
    return c


def validate(c):
    row=next((r for r in rows() if r['row_id']==c.get('row_id')),None)
    if row is None or c!=config(row):raise ValueError('Changed/unregistered dual-evidence row')
    return c


def make_args(c,device='cuda:0'):
    validate(c);a=args_from_validated_config(c,device)
    a.pseudo_domain_gate=False;a.pseudo_temporal_mode='batch_neighbor';a.pseudo_temporal_bank_min_streak=2
    a.phase1_lr_schedule='cosine';a.phase1_lr_min=1e-6
    return a
