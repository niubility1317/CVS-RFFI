import math
import sys
from pathlib import Path
PROJECT=Path('/home/szu2070436088/2510044040/CV-SincNet')
WORKER=PROJECT/'releases/cvs_reference_repair_20261008_r01'
sys.path[:0]=[str(WORKER),str(WORKER/'code')]
WORKER_COMMIT='ae783c81cd4949f93a80dbd2e80c4b2568eb99ea'
RUN='20261009-phase1-ce-singlepass-batch-manysig-m4-r03'
BASE=PROJECT/'runs'/RUN
BATCHES=(256,512,1024,2048)
SEED=2026092701
def config(batch):
 from experiments.cvs_phase1_repair import design as d
 c=d.config(dict(stage='repair',arm='ce_cosine',model_seed=SEED,row_id='ce_cosine-s'+str(SEED)))
 rid='ce_b'+str(batch)+'-s'+str(SEED)
 c.update(run_id=RUN,row_id=rid,arm='ce_b'+str(batch),stage='ce_batch',output_root=(BASE/rid/'source').as_posix(),batch_size=batch,steps_per_epoch=math.ceil(6300/batch),epoch_mode='labeled_single_pass',labeled_drop_last=False)
 return c
def validate(c):
 if c.get('batch_size') not in BATCHES or c!=config(c['batch_size']):raise ValueError('Changed fixed CE single-pass configuration')
 return c
def make_args(c,device='cuda:0'):
 validate(c)
 from experiments.cvs_phase1_stack.design import args_from_validated_config
 a=args_from_validated_config(c,device);a.batch_size=c['batch_size'];a.phase1_lr_schedule='cosine';a.phase1_lr_min=1e-6
 a.pseudo_domain_gate=False
 return a
def require_budget(logged,successful):
 if logged!=successful or logged not in {200*math.ceil(6300/b) for b in BATCHES}:raise ValueError('Incomplete single-pass budget')
