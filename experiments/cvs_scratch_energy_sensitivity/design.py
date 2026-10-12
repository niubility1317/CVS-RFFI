"""Fixed layer scales retain packet-to-packet margin/CE sensitivity amplitude."""
from copy import deepcopy
from pathlib import Path
from experiments.cvs_scratch_identity_style import design as old
ROOT=Path(__file__).resolve().parents[2]
PROJECT,SOURCE,CAPSULE,TRUTH,CLASSES,SEEDS,FULL_FP32_POLICY=(getattr(old,k) for k in ('PROJECT','SOURCE','CAPSULE','TRUTH','CLASSES','SEEDS','FULL_FP32_POLICY'))
read,write,baseline=old.read,old.write,old.baseline
RUN='20261012-phase1-scratch-energy-sensitivity-manysig-m6-r01'
RELEASE='cvs_scratch_energy_sensitivity_20261012_r01'
OWNER='codex/root/scratch-energy-sensitivity-20261012'
BASE=Path(PROJECT)/'runs'/RUN
ARMS=('MARGIN_ENERGY','CE_ENERGY')
PRIMARY='CE_ENERGY'
PAIRS=[('CE_ENERGY','MARGIN_ENERGY')]+[(a,'P0') for a in ARMS]
REFERENCE_RUN=old.RUN
REFERENCE_PAIRS=[(a,'DELAYCAP_JOINT') for a in ARMS]

def rows():return [dict(arm=a,model_seed=s,row_id=f'{a}-s{s}') for s in SEEDS for a in ARMS]

def config(row):
    if row not in rows():raise ValueError('Unregistered energy-sensitivity row')
    prior=next(r for r in old.rows() if r['arm']=='DELAYCAP_JOINT' and r['model_seed']==row['model_seed'])
    c=deepcopy(old.config(prior))
    c.update(schema='scratch_energy_sensitivity_v1',run_id=RUN,**row,method='cvs_scratch_energy_sensitivity',stage='phase1_scratch_energy_sensitivity',
        output_root=(BASE/row['row_id']/'source').as_posix(),launch_owner=OWNER)
    c['teacher']['cache_root']=(BASE/'peer_cache').as_posix()
    c['multilevel'].update(statistic='raw_sensitivity',sensitivity_objective='ce' if row['arm']=='CE_ENERGY' else 'margin',
        sensitivity_normalization='fixed_layer_source_RMS',scale_calibration_pairs=12,scale_calibration_readouts=1440,
        calibration_total_forwards=48,calibration_total_readouts=2880,scale_after_identity_updates=444,
        fishr_estimator='finite_L_corrected_cross_AB_energy_sensitivity')
    return c

def validate(c):
    row=next((r for r in rows() if r['row_id']==c.get('row_id')),None)
    if row is None or c!=config(row):raise ValueError('Energy-sensitivity configuration differs')
    return c

def learning_rate(c,epoch):return old.learning_rate(c,epoch)
