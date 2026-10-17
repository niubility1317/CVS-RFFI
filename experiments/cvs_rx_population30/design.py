"""One finite-budget As receiver-population coverage hypothesis; fixed existing nominal parent."""
from copy import deepcopy
from pathlib import Path
from experiments.cvs_clock_factorial30 import design as trained
ROOT=Path(__file__).resolve().parents[2]
RUN='20261017-phase1-F-crossRX-RXpopulation-e20plus10-s2026092701-m1-r01';RELEASE='cvs_F_RXpopulation_E30_20261017_r01';LOCAL='local_artifacts/F_RXpopulation_E30_20261017';OWNER='codex/root/F-RXpopulation-E30'
read,write=trained.read,trained.write
PROJECT,SOURCE,CAPSULE,TRUTH,VIEWS_ROOT=trained.PROJECT,trained.SOURCE,trained.CAPSULE,trained.TRUTH,trained.VIEWS_ROOT
AF_RELEASE,SEED,FIT,SCENES,VIEWS,CLASSES,FULL_FP32_POLICY=trained.AF_RELEASE,trained.SEED,trained.FIT,trained.SCENES,trained.VIEWS,trained.CLASSES,trained.FULL_FP32_POLICY
TRAIN_SCENES,CHANNEL_SEEDS=trained.TRAIN_SCENES,trained.CHANNEL_SEEDS
POPULATION_SEED=CHANNEL_SEEDS['receiver']
STEPS,UPDATES=222,2220;BASE=Path(PROJECT)/'runs'/RUN
CONTROL_RUN='20261017-phase1-F-crossRX-clock-e30-score-recovery-s2026092701-m5-r01';CONTROL_BASE=Path(PROJECT)/'runs'/CONTROL_RUN/'FULL_AS-s2026092701'
def runtime():
 p=ROOT/'runtime.json';return read(p if p.is_file() else ROOT/LOCAL/'runtime.json')
def rows():return [dict(row_id='RX_POPULATION-s2026092701',arm='FULL_AS',model_seed=SEED)]
def train_rows():return rows()
def nominal_config():return trained.config(next(r for r in trained.rows() if r['arm']=='FULL_AS'))
def parent_config(arm):
 if arm!='FULL_AS':raise ValueError('Fixed ordinary As parent only')
 return trained.parent_config(arm)
parent_source=trained.parent_source
parent_proof=trained.parent_proof;validate_parent_checkpoint=trained.validate_parent_checkpoint;base_model_config=trained.base_model_config

def config(row):
 if row not in rows():raise ValueError('Only fixed RX population variant')
 c=deepcopy(nominal_config());c.update(run_id=RUN,**row,launch_owner=OWNER,output_root=(BASE/row['row_id']/'source').as_posix(),schema='F_PRACTICAL_RX_POPULATION_E20PLUS10_V1',model_initialization='strict own source-final20 plus ownEMA;optimizer reset',checkpoint_sources=[(parent_source('FULL_AS')/'final_ssdg.pth').as_posix()],parent_selection='Same ordinaryAs source-final20 as fixed nominal E30;not target-ranked',training_recipe='F+CROSS_RX originalU/defaulthead;E20+10/cos10-reset;full virtualRX session per epoch',population_mode='epoch_session_original_receiver_prior',population_seed=POPULATION_SEED,population_policy=dict(real_source_RX='all original5 unchanged;not replaced',virtual_hardware='one original receiver_for_session draw per RX/day/eq/cumulativeepoch,stable within virtualsession',true_amplitude_max_db=.2,true_quadrature_max_deg=1.,receiver_cfo_std_hz=200.,amp_error_std_db=.05,phase_error_std_deg=.2,estimated_IQ='truth plus original independent Gaussian errors',scope='150 virtualsessions across cumulative21..30;no packetwise hardware redraw',TX_dependence=False,source_of_scales='unchanged existing simulator Config,not source/target fitted'),control_run=CONTROL_RUN,control_row='FULL_AS-s2026092701',physical_channel_prepasses=0,extra_physical_filters=0,new_neural_forwards=0,new_losses=0,new_U_rules=0,new_Ag=False,As=True,budget_authorization='One fixed ordinaryAs parentE20+10=E30;2220new/6660cumulativeupdates',cost_claim='L1136640NN/U567000 unchanged;150 virtualRX draws;0new NN/fullchannelprepass;teacher/eval/wall counted',source_selection='Fixed final10/cumulative30;allsourceRX,no source holdout or target selection',selection='fixed final10/cumulative30;no targetfeedback or selective rerun',scientific_falsification=dict(low_urban_delta_vs_nominal_min_pp=.5,clean_delta_vs_nominal_min_pp=0.,midurban_delta_vs_nominal_min_pp=0.,each_suburban_delta_min_pp=-.25,all_joint_targets_also_reported=True,no_automatic_seed_extension=True))
 return c

def validate(c):
 row=next((r for r in rows() if r['row_id']==c.get('row_id')),None)
 if row is None or c!=config(row):raise ValueError('Fixed population/data/parent/E30 scope differs')
 return c
def make_args(c,device='cuda:0'):
 validate(c);a=trained.make_args(nominal_config(),device);a.run_id=RUN;a.candidate_id=c['row_id'];a.output_dir=c['output_root'];return a
