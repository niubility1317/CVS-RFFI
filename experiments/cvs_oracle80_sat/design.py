"""Matched scratch E30 full-label reference with LEO weak training views."""
from copy import deepcopy
from pathlib import Path
from experiments.cvs_oracle80 import design as control

ROOT=Path(__file__).resolve().parents[2]
PROJECT=control.PROJECT
RUN='20261012-phase1-oracle80-sat-e30-manysig-s2026092701-r01'
RELEASE='cvs_oracle80_sat_e30_20261012_r01'
BASE=Path(PROJECT)/'runs'/RUN
SOURCE=control.SOURCE  # Identical physical roles: reuse VALIDATED_ONCE.
CAPSULE=control.CAPSULE;TRUTH=control.TRUTH
CLASSES=control.CLASSES;FULL_FP32_POLICY=control.FULL_FP32_POLICY
SEEDS=control.SEEDS;OWNER='codex/root/oracle80-sat-e30-20261012'
read,write=control.read,control.write
SCENES=control.SCENES;VIEWS=control.VIEWS;VIEWS_ROOT=control.VIEWS_ROOT;COUNT=control.COUNT
TRAIN_SCENES=('leo_clear_weak','leo_low_elev_weak','leo_rain_weak')
SCHEDULE='1@0.30:leo_clear_weak;7@0.60:leo_low_elev_weak,leo_rain_weak;15@0.80:leo_clear_weak,leo_low_elev_weak,leo_rain_weak'
SAT_ARGS=dict(use_concat_sat_channel_aug=True,concat_sat_ce_only=True,concat_sat_fused_ce_only=True,
    sat_training_mode='concat_masked',lambda_sat_cls=.68,lambda_sat_cons=0.,use_sat_consistency=False,
    concat_sat_start_epoch=1,sat_cons_start_epoch=1,concat_sat_ce_weight=1.,sat_view_seed=2027,
    sat_train_scenario='leo_clear_weak',sat_train_scenarios=','.join(TRAIN_SCENES),sat_view_schedule=SCHEDULE,sat_view_prob=1.)

def rows():return control.rows()

def config(row):
    c=deepcopy(control.config(row));c.update(schema='oracle80_sat_e30_v1',stage='oracle80_sat_e30',run_id=RUN,
        method='cvs_oracle80_sat',output_root=(BASE/row['row_id']/'source').as_posix(),features=['leo'],launch_owner=OWNER,
        training_augmentation=dict(satellite_ce_weight=.68,consistency_weight=0.,start_epoch=1,view_seed=2027,
            scenarios=list(TRAIN_SCENES),schedule=SCHEDULE,mode='concat_masked',fused_clean_satellite=True,
            probability_scope='per batch; not selected batches use duplicate clean view with same auxiliary CE weight',
            compression='E200 view stages40/90 compressed to6/14; satellite CE beginsE1 for actualE30 exposure'),
        control_run_id=control.RUN,control_prediction=(control.BASE/row['row_id']/'prediction/predictions.npz').as_posix())
    return c

def validate(c):
    if c!=config(rows()[0]):raise ValueError('Satellite oracle configuration changed')
    return c

def make_args(c,device='cuda:0'):
    validate(c);a=control.make_args(control.config(rows()[0]),device)
    a.output_dir=c['output_root'];a.candidate_id=c['row_id'];a.run_id=c['run_id']
    for key,value in SAT_ARGS.items():setattr(a,key,value)
    from experiments.cvs_phase1_overlay.model import native_modules
    native_modules()._resolve_sat_training_mode(a)
    return a
