"""Fixed cross-view dropout coupling, two matched full-source E20+10 rows."""
from copy import deepcopy
from pathlib import Path
from experiments.cvs_clock_factorial30 import design as trained
ROOT=Path(__file__).resolve().parents[2]
RUN='20261017-phase1-F-crossRX-dropoutCoupling-e20plus10-s2026092701-m2-r01';RELEASE='cvs_F_dropoutCoupling_E30_20261017_r01';LOCAL='local_artifacts/F_dropoutCoupling_E30_20261017';OWNER='codex/root/F-dropoutCoupling-E30'
read,write=trained.read,trained.write
PROJECT,SOURCE,CAPSULE,TRUTH,VIEWS_ROOT=trained.PROJECT,trained.SOURCE,trained.CAPSULE,trained.TRUTH,trained.VIEWS_ROOT
AF_RELEASE,SEED,FIT,SCENES,VIEWS,CLASSES,FULL_FP32_POLICY=trained.AF_RELEASE,trained.SEED,trained.FIT,trained.SCENES,trained.VIEWS,trained.CLASSES,trained.FULL_FP32_POLICY
TRAIN_SCENES,CHANNEL_SEEDS=trained.TRAIN_SCENES,trained.CHANNEL_SEEDS
STEPS,UPDATES,MASK_SEED=222,2220,20261017561;BASE=Path(PROJECT)/'runs'/RUN
MODES=('INDEPENDENT','ANTITHETIC')
def runtime():
 p=ROOT/'runtime.json';return read(p if p.is_file() else ROOT/LOCAL/'runtime.json')
def rows():return [dict(row_id=mode+'-s'+str(SEED),arm='FULL_AS',model_seed=SEED,mask_mode=mode) for mode in MODES]
def train_rows():return rows()
def nominal_config():return trained.config(next(r for r in trained.rows() if r['arm']=='FULL_AS'))
def parent_config(arm):
 if arm!='FULL_AS':raise ValueError('Fixed ordinary As parent only')
 return trained.parent_config(arm)
parent_source=trained.parent_source
parent_proof=trained.parent_proof;validate_parent_checkpoint=trained.validate_parent_checkpoint;base_model_config=trained.base_model_config

def config(row):
 if row not in rows():raise ValueError('Only fixed paired dropout matrix')
 c=deepcopy(nominal_config());c.update(run_id=RUN,**row,launch_owner=OWNER,output_root=(BASE/row['row_id']/'source').as_posix(),schema='F_DROPOUT_COUPLING_E20PLUS10_V1',model_initialization='strict own source-final20 plus ownEMA;optimizer reset',checkpoint_sources=[(parent_source('FULL_AS')/'final_ssdg.pth').as_posix()],parent_selection='Original complete-source ordinaryAs fixed source-final20;not target-ranked',training_recipe='F+CROSS_RX originalU/defaulthead;E20+10/cos10-reset;cross-view dropout coupling',dropout_policy=dict(seed=MASK_SEED,integer_uniform_size=2**24,view_order=['O1','O2','S1','S2'],scope='existing identity encoder nn.Dropout only;other modules native',ordinary_pair_independent=True,As_pair_independent=True,matching='S1 vs O1; S2 vs O2',marginal='same engine keep threshold round((1-p)*2^24),scale1/(1-p);native p unchanged',draws='four independent private uniform tensors per active dropout call in both rows;unused S tensors still drawn in ANTITHETIC',private_key='fixedseed,step,moduleindex,occurrence;no IQ,TX,RX,day',U_teacher='native dropout outside armed L scope;global RNG untouched by L engine',losses='unchanged .5 ordinaryRD + .5 AsRD + .1 original common SupCon;both KL kept',claim='training organization only;not IQ Ag or known identity improvement'),physical_channel_prepasses=0,extra_physical_filters=0,new_neural_forwards=0,new_losses=0,new_U_rules=0,new_Ag=False,As=True,IQ_context=dict(length=256,longer_context_mapping_available=False),budget_authorization='Two fixed same ordinaryAs parentE20+10 rows;E30 per model,2220new/6660cumulativeupdates each',cost_claim='Same4Lforwards each;private mask generation+cache cost matched and counted;teacher/eval/parent separately',source_selection='fixed final10/cumulative30;allsourceRX;no heldout trainingRX or targetselection',selection='fixed final10/cumulative30;no targetfeedback or selective rerun',scientific_falsification=dict(low_urban_delta_vs_independent_min_pp=.5,clean_delta_vs_independent_min_pp=0.,midurban_delta_vs_independent_min_pp=0.,each_suburban_delta_min_pp=-.25,all_joint_targets_also_reported=True,no_automatic_seed_extension=True,no_mask_key_or_probability_rescue=True))
 return c

def validate(c):
 row=next((r for r in rows() if r['row_id']==c.get('row_id')),None)
 if row is None or c!=config(row):raise ValueError('Fixed mask/data/parent/E30 scope differs')
 return c
def make_args(c,device='cuda:0'):
 validate(c);a=trained.make_args(nominal_config(),device);a.run_id=RUN;a.candidate_id=c['row_id'];a.output_dir=c['output_root'];return a
