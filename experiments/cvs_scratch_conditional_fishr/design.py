"""Source-driven fixed conditional interaction matrix, frozen before target readback."""
from copy import deepcopy
from pathlib import Path
from experiments.cvs_scratch_energy_sensitivity import design as old
ROOT=Path(__file__).resolve().parents[2]
PROJECT,SOURCE,CAPSULE,TRUTH,CLASSES,SEEDS,FULL_FP32_POLICY=(getattr(old,k) for k in ('PROJECT','SOURCE','CAPSULE','TRUTH','CLASSES','SEEDS','FULL_FP32_POLICY'))
read,write,baseline=old.read,old.write,old.baseline
RUN='20261012-phase1-scratch-conditional-fishr-manysig-m9-r01'
RELEASE='cvs_scratch_conditional_fishr_20261012_r01'
OWNER='codex/root/scratch-conditional-fishr-20261012'
BASE=Path(PROJECT)/'runs'/RUN
ARMS=('MARGIN_INTERACTION','CE_INTERACTION','CE_MOMENT_INTERACTION')
PRIMARY='CE_MOMENT_INTERACTION'
PAIRS=[('CE_INTERACTION','MARGIN_INTERACTION'),('CE_MOMENT_INTERACTION','CE_INTERACTION')]+[(a,'P0') for a in ARMS]
REFERENCE_RUN=old.RUN
REFERENCE_PAIRS=[('MARGIN_INTERACTION','MARGIN_ENERGY'),('CE_INTERACTION','CE_ENERGY'),('CE_MOMENT_INTERACTION','CE_ENERGY')]
ESTIMATOR='independent_AB_TX_RX_interaction_and_finite_L_moment'

def rows():return [dict(arm=a,model_seed=s,row_id=f'{a}-s{s}') for s in SEEDS for a in ARMS]

def config(row):
    if row not in rows():raise ValueError('Unregistered conditional row')
    parent_arm='MARGIN_ENERGY' if row['arm']=='MARGIN_INTERACTION' else 'CE_ENERGY'
    prior=next(r for r in old.rows() if r['arm']==parent_arm and r['model_seed']==row['model_seed'])
    c=deepcopy(old.config(prior))
    c.update(schema='scratch_conditional_fishr_v1',run_id=RUN,**row,method='cvs_scratch_conditional_fishr',stage='phase1_scratch_conditional_fishr',
        output_root=(BASE/row['row_id']/'source').as_posix(),launch_owner=OWNER)
    c['teacher']['cache_root']=(BASE/'peer_cache').as_posix()
    c['optimizer']['schedule']='fresh AdamW; epoch cosine20; same LR for all identity parameters; random student total20; independent pretrained teacher cost separate'
    c['multilevel'].update(fishr_estimator=ESTIMATOR,
        fishr_scope='main CE/margin affine sensitivity divided by fixed source layer RMS; double-centered TX/RX conditional AB cross moment; optional finite-L corrected mixture variance',
        gradient_budget_scope='after444, separate 12-pair calibration per component; equal component/layer means then one detached complete-identity cap0.2',
        aux_ce_routing='no learned auxiliary heads; zero auxiliary CE; legacy auxiliary CE weight inactive',
        routing_control='no warmup heads or auxiliary optimizer; live mixed second derivatives every4 after444; eval captures preserve identity RNG and module modes',
        conditional_components=['moment','interaction'] if row['arm']=='CE_MOMENT_INTERACTION' else ['interaction'],
        conditional_reduction='equal mean of separately gradient-calibrated components; layer mean /2 then one shared joint cap',
        conditional_grid=dict(TX=list(range(6)),RX=[1,3,4,6,8],per_cell=2,same_day=True),
        conditional_statistic='double-centered TX/RX conditional sensitivity means; independent AB inner product; signed')
    return c

def validate(c):
    row=next((r for r in rows() if r['row_id']==c.get('row_id')),None)
    if row is None or c!=config(row):raise ValueError('Conditional Fishr configuration differs')
    return c

def learning_rate(c,epoch):return old.learning_rate(c,epoch)
