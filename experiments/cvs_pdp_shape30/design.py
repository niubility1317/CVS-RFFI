"""One fixed RMS-matched propagation-profile mixture;own E20 parent plus10."""
from copy import deepcopy
from pathlib import Path
from experiments.cvs_clock_factorial30 import design as trained
ROOT=Path(__file__).resolve().parents[2]
RUN='20261017-phase1-F-crossRX-PDPshape-e20plus10-s2026092701-m1-r01'
RELEASE='cvs_F_PDPshape_E30_20261017_r01'
LOCAL='local_artifacts/F_pdp_shape_E30_20261017'
OWNER='codex/root/F-PDPshape-E30'
read,write=trained.read,trained.write
PROJECT,SOURCE,CAPSULE,TRUTH,VIEWS_ROOT=trained.PROJECT,trained.SOURCE,trained.CAPSULE,trained.TRUTH,trained.VIEWS_ROOT
AF_RELEASE,SEED,FIT,SCENES,VIEWS,CLASSES,FULL_FP32_POLICY=trained.AF_RELEASE,trained.SEED,trained.FIT,trained.SCENES,trained.VIEWS,trained.CLASSES,trained.FULL_FP32_POLICY
TRAIN_SCENES,CHANNEL_SEEDS=trained.TRAIN_SCENES,trained.CHANNEL_SEEDS
PROFILE_SEED=20261017711
STEPS,UPDATES=222,2220
BASE=Path(PROJECT)/'runs'/RUN
CONTROL_RUN='20261017-phase1-F-crossRX-clock-e30-score-recovery-s2026092701-m5-r01'
CONTROL_BASE=Path(PROJECT)/'runs'/CONTROL_RUN/'FULL_AS-s2026092701'
def runtime():
 p=ROOT/'runtime.json';return read(p if p.is_file() else ROOT/LOCAL/'runtime.json')
def rows():return [dict(row_id='PDP_SHAPE-s2026092701',arm='FULL_AS',model_seed=SEED)]
def train_rows():return rows()
def nominal_config():return trained.config(next(r for r in trained.rows() if r['arm']=='FULL_AS'))
def parent_config(arm):
 if arm!='FULL_AS':raise ValueError('Original ordinary As source-final20 only')
 return trained.parent_config(arm)
parent_source=trained.parent_source
parent_proof=trained.parent_proof
validate_parent_checkpoint=trained.validate_parent_checkpoint
base_model_config=trained.base_model_config

def config(row):
 if row not in rows():raise ValueError('Single declared profile mixture only')
 c=deepcopy(nominal_config())
 c.update(run_id=RUN,**row,launch_owner=OWNER,output_root=(BASE/row['row_id']/'source').as_posix(),schema='F_PRACTICAL_PDP_SHAPE_E20PLUS10_V1',model_initialization='strict own source-final20 plus ownEMA;optimizer reset',checkpoint_sources=[(parent_source('FULL_AS')/'final_ssdg.pth').as_posix()],parent_selection='Same ordinaryAs source-final20 as nominal E30;not target-ranked',training_recipe='F+CROSS_RX originalU/defaulthead;E20+10/cos10-reset;one propagation delay-shape mixture',profile_seed=PROFILE_SEED,profile_policy=dict(shapes=[[0.,1.,3.],[0.,1.,2.]],probabilities=[.5,.5],sampling='one private physical-sample/step draw;no TX labels',RMS='exact original total ensemble PDP including mean LOS power per state',preserved='original diffuse tap powers,LOS shadow/state/geometry/rays/noise/RX session/calibration/sync/tracking/AGC law/reflect/common24 latency',changed='diffuse path delay locations;H-dependent quality and receiver processing recomputed',amplitude_renormalization=False,source_of_shapes='existing engineering three-ray proxy and equispaced alternative;not a claimed 3GPP TDL profile or empirical fit'),control_run=CONTROL_RUN,control_row='FULL_AS-s2026092701',physical_channel_prepasses=0,extra_physical_filters=0,new_neural_forwards=0,new_losses=0,new_U_rules=0,new_Ag=False,As=True,budget_authorization='one fixed ordinaryAs parentE20+10=E30;2220new/6660cumulativeupdates',cost_claim='same four L forwards;one complete residual channel per As;profile algebra only;teacher/eval/wall separately counted',source_selection='fixed final10/cumulative30;all five sourceRX;no source holdout',selection='fixed final10/cumulative30;no targetfeedback or selective rerun',scientific_falsification=dict(low_urban_delta_vs_nominal_min_pp=.5,clean_delta_vs_nominal_min_pp=0.,midurban_delta_vs_nominal_min_pp=0.,each_suburban_delta_min_pp=-.25,all_joint_targets_also_reported=True,no_automatic_seed_extension=True))
 return c

def validate(c):
 row=next((r for r in rows() if r['row_id']==c.get('row_id')),None)
 if row is None or c!=config(row):raise ValueError('Fixed PDP/data/parent/E30 contract differs')
 return c

def make_args(c,device='cuda:0'):
 validate(c);a=trained.make_args(nominal_config(),device);a.run_id=RUN;a.candidate_id=c['row_id'];a.output_dir=c['output_root'];return a
