"""Predeclared fixed comparison of complementary disentanglement constraints."""
from pathlib import Path
from experiments.cvs_phase1_repair import design as baseline
from experiments.cvs_phase1_stack.source import require_budget
from experiments.cvs_phase1_stack.design import PROJECT,SOURCE,CAPSULE,TRUTH,SEEDS,CLASSES,FULL_FP32_POLICY,read,write,args_from_validated_config
from experiments.cvs_multi_disentangle.physics import contract as intervention_contract

ROOT=Path(__file__).resolve().parents[2]
RUN='20261009-phase1-multi-disentangle-manysig-m32-r01'
RELEASE='cvs_multi_disentangle_20261009_r01'
BASE=Path(PROJECT)/'runs'/RUN
OWNER='codex/root/multi-disentangle-20261009'
ARMS=('legacy_cosine','fixed_interventions','unified','linear','temporal','receiver','multi','multi_interaction')
RECIPE=dict(warmup_epochs=20,auxiliary_every_steps=4,auxiliary_batch_size=32,
    rank=8,code_dim=8,auxiliary_lr=.0002,auxiliary_lr_min=.000001,
    identity_aux_weight=.05,paired_consistency_weight=.03,receiver_identity_weight=.01,
    residual_bound=.25,receiver_min_group_count=2,receiver_max_age_steps=888,
    receiver_relations_max=8,receiver_quality_thresholds=[.25,.75],
    receiver_response_rms_threshold=1.,receiver_abs_cfo_threshold=.1,
    auxiliary_data='labeled_source_train_only',
    deployment='identity_only',resume=False,source_selection='fixed_E200_all_controls',
    interactions='explicit_L_then_T_factorial',interventions=intervention_contract(),
    randomness='private_generator_no_native_RNG_draws',
    receiver_scope='quality_matched_group_statistics_not_packet_counterfactual')

def rows():
    return [dict(stage='multi_disentangle',arm=a,model_seed=s,row_id=a+'-s'+str(s)) for s in SEEDS for a in ARMS]

def config(row):
    c=baseline.config(dict(stage='repair',arm='pseudo_batch_cosine',model_seed=row['model_seed'],row_id='pseudo_batch_cosine-s'+str(row['model_seed'])))
    c.update(schema='multi_disentangle_v1',run_id=RUN,**row,method='cvs_multi_disentangle',
        output_root=(BASE/row['row_id']/'source').as_posix(),multi_recipe=RECIPE,
        baseline_recipe='cvs_phase1_repair/pseudo_batch_cosine',checkpoint_sources=[])
    return c

def validate(c):
    row=next((r for r in rows() if r['row_id']==c.get('row_id')),None)
    if row is None or c!=config(row):raise ValueError('Changed/unregistered multi-disentangle row')
    return c

def make_args(c,device='cuda:0'):
    validate(c);a=args_from_validated_config(c,device)
    a.pseudo_domain_gate=False;a.pseudo_temporal_mode='batch_neighbor';a.pseudo_temporal_bank_min_streak=2
    a.phase1_lr_schedule='cosine';a.phase1_lr_min=1e-6
    return a
