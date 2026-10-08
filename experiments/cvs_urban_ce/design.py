"""User-fixed CE + urban channel/DG contrasts. No target-derived selection."""
from pathlib import Path
from experiments.cvs_phase1_stack.design import (
    PROJECT,SOURCE,CAPSULE,TRUTH,SEEDS,CLASSES,FULL_FP32_POLICY,read,write,args_from_validated_config)
from experiments.cvs_phase1_repair.design import cosine_lr
ROOT=Path(__file__).resolve().parents[2]
RUN='20261008-phase1-urban-ce-manysig-m24-r01'
RELEASE='cvs_urban_ce_20261008_r01'
BASE=Path(PROJECT)/'runs'/RUN
OWNER='codex/root/urban-ce-20261008'
ARMS=('ce','legacy_leo','urban','receiver_dg','urban_dg','urban_dg_thin')
URBAN_SCHEDULE='1@0.30:practical_high_urban;41@0.60:practical_mid_urban,practical_low_urban;91@0.80:practical_mid_urban,practical_low_urban'

def rows():
    return [dict(stage='urban_ce',arm=a,model_seed=s,row_id=a+'-s'+str(s)) for s in SEEDS for a in ARMS]

def config(row):
    arm=row['arm']
    return dict(schema='urban_ce_v1',run_id=RUN,**row,features=['leo'] if arm not in ('ce','receiver_dg') else [],
        variant='reference_response',source_contract=SOURCE,dataset=PROJECT+'/Dataset_WigSig/ManySig.pkl',
        output_root=(BASE/row['row_id']/'source').as_posix(),epochs=200,steps_per_epoch=222,
        model_initialization='scratch',checkpoint_sources=[],numerical_policy=FULL_FP32_POLICY,
        method='cvs_urban_ce',target_access=False,lr_schedule='cosine',lr_min=1e-6,
        urban=arm.startswith('urban'),receiver_dg=arm in ('receiver_dg','urban_dg','urban_dg_thin'),
        echo_l1_max=.15,echo_delays=[1,2,4],dg_probability=.5,dg_ramp_epochs=40,
        satellite_compute_probability=.5 if arm=='urban_dg_thin' else 1.,
        augmentation_seed=row['model_seed']+17001,channel_seed=2027,
        extra_losses=[],selection='ALL_PREREGISTERED_FIXED_CONTROLS')

def validate(c):
    row=next((r for r in rows() if r['row_id']==c.get('row_id')),None)
    if row is None or c!=config(row):raise ValueError('Unregistered or modified urban CE row')
    return c

def make_args(c,device='cuda:0'):
    validate(c);a=args_from_validated_config(c,device)
    a.phase1_lr_schedule='cosine';a.phase1_lr_min=c['lr_min']
    # No fallback random satellite forward when the sampled CE view is absent.
    # The explicit concat CE branch is independent of use_sat_consistency.
    a.use_sat_consistency=False
    a.sat_cons_start_epoch=80
    if c['urban']:a.sat_view_schedule=URBAN_SCHEDULE
    return a
