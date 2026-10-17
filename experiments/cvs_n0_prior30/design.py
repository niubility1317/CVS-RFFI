"""One fixed linear-N0 prior;own ordinaryAs E20 parent plus10."""
from copy import deepcopy
from pathlib import Path
from experiments.cvs_clock_factorial30 import design as trained
ROOT=Path(__file__).resolve().parents[2]
RUN='20261017-phase1-F-crossRX-N0prior-e20plus10-s2026092701-m1-r01'
RELEASE='cvs_F_N0prior_E30_20261017_r01'
LOCAL='local_artifacts/F_N0_prior_E30_20261017'
OWNER='codex/root/F-N0prior-E30'
read,write=trained.read,trained.write
PROJECT,SOURCE,CAPSULE,TRUTH,VIEWS_ROOT=trained.PROJECT,trained.SOURCE,trained.CAPSULE,trained.TRUTH,trained.VIEWS_ROOT
AF_RELEASE,SEED,FIT,SCENES,VIEWS,CLASSES,FULL_FP32_POLICY=trained.AF_RELEASE,trained.SEED,trained.FIT,trained.SCENES,trained.VIEWS,trained.CLASSES,trained.FULL_FP32_POLICY
TRAIN_SCENES,CHANNEL_SEEDS=trained.TRAIN_SCENES,trained.CHANNEL_SEEDS
PROCESSING_SEED=20261017961
STEPS,UPDATES=222,2220
BASE=Path(PROJECT)/'runs'/RUN
CONTROL_RUN='20261017-phase1-F-crossRX-clock-e30-score-recovery-s2026092701-m5-r01'
CONTROL_BASE=Path(PROJECT)/'runs'/CONTROL_RUN/'FULL_AS-s2026092701'
def runtime():
 p=ROOT/'runtime.json';return read(p if p.is_file() else ROOT/LOCAL/'runtime.json')
def rows():return [dict(row_id='N0_PRIOR-s2026092701',arm='FULL_AS',model_seed=SEED)]
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
 if row not in rows():raise ValueError('Single fixed linear-N0 prior only')
 c=deepcopy(nominal_config())
 c.update(run_id=RUN,**row,launch_owner=OWNER,output_root=(BASE/row['row_id']/'source').as_posix(),schema='F_PRACTICAL_N0_PRIOR_E20PLUS10_V1',model_initialization='strict own source-final20 plus ownEMA;optimizer reset',checkpoint_sources=[(parent_source('FULL_AS')/'final_ssdg.pth').as_posix()],parent_selection='Same ordinaryAs source-final20 as nominalE30;no target ranking',training_recipe='F+CROSS_RX originalU/defaulthead;E20+10/cos10-reset;one equal-prior linear-N0 multiplier;original quality and processing recomputed',processing_seed=PROCESSING_SEED,noise_policy=dict(factors=[.5,1.5],probabilities=[.5,.5],sampling='one private physical-sample/step draw;no TX label',nominal_reference_snr_db=30.,nominal_linear_N0=.001,expected_linear_N0=.001,realized_mean_exactly_nominal=False,quality='recompute original receiver quality/lock/tracking/residual scale/AGC under selected N0',preserved='Original scene/H/state/shadow/geometry/scatter/standard noise/session hardware/calibration priors and reflect',changed='One original noise injection variance .001 times factor;original quality dependencies retained',equalization_enabled=False,amplitude_renormalization=False,range_origin='Fixed engineering prior from previous source study;not measured temperature or target-selected',previous_source_study='20261017-diagnostic-F-N0prior-fullL-s2026092701-m1-r01',previous_source_result='FAIL_FIXED_N0_PRIOR_SOURCE_CANDIDATE_CLOSED',training_question='Previous frozen input criterion failed;no training utility was tested. First actual training test;no source rerun, factor/probability/threshold rescue or passed-source claim'),control_run=CONTROL_RUN,control_row='FULL_AS-s2026092701',physical_channel_prepasses=0,new_neural_forwards=0,new_losses=0,new_U_rules=0,new_Ag=False,As=True,budget_authorization='one ordinaryAs ownparent E20plus10 cumulativeE30;2220new/6660cumulativeupdates',cost_claim='Same4L neuralforwards;one complete residual perAs;actual CPU/wall separatelycounted',source_selection='fixedfinal10/cumulative30;all5sourceRX;no sourceholdout',selection='fixedfinal10/cumulative30;no targetfeedback/selective rerun',scientific_falsification=dict(low_urban_delta_vs_nominal_min_pp=.5,clean_delta_vs_nominal_min_pp=0.,midurban_delta_vs_nominal_min_pp=0.,each_suburban_delta_min_pp=-.25,all_joint_targets_also_reported=True,no_automatic_seed_extension=True))
 c['channel']['EQ']=False
 c['channel']['N0_prior']='Equalprior .5/1.5 times nominal linearN0;coupled quality recomputed;trainL only'
 return c

def validate(c):
 row=next((r for r in rows() if r['row_id']==c.get('row_id')),None)
 if row is None or c!=config(row):raise ValueError('Fixed N0/data/parent/E30 contract differs')
 return c

def make_args(c,device='cuda:0'):
 validate(c);a=trained.make_args(nominal_config(),device);a.run_id=RUN;a.candidate_id=c['row_id'];a.output_dir=c['output_root'];return a
