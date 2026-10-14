from pathlib import Path
from experiments.cvs_feature_mean_teacher50 import design as first
from experiments.cvs_feature_mean_teacher_seeds import design as remaining
ROOT=Path(__file__).resolve().parents[2]
PROJECT,SOURCE,FULL_FP32_POLICY,read,write,PARAMETERS=(getattr(first,k) for k in ('PROJECT','SOURCE','FULL_FP32_POLICY','read','write','PARAMETERS'))
RUN='20261015-phase1-source-feature-ce-gradient-manysig-m6-r01'
RELEASE='cvs_source_feature_ce_gradient_20261015_r01';OWNER='codex/root/source-feature-ce-gradient-20261015'
BASE=Path(PROJECT)/'runs'/RUN
SEEDS=(2026092701,2026092702,2026092703);ARMS=('SIG_TIME','FEATURE_MT')
GROUPS=('WEAK_RIGHT_STRONG_RIGHT','WEAK_RIGHT_STRONG_WRONG','WEAK_WRONG_STRONG_RIGHT','WEAK_WRONG_STRONG_WRONG')
def rows():return [dict(row_id=arm+'-s'+str(seed),model_seed=seed,arm=arm) for seed in SEEDS for arm in ARMS]
def prior(row):return first if row['model_seed']==SEEDS[0] else remaining
def source_config(row):
 p=prior(row);return p.config(next(r for r in p.rows() if r['arm']==row['arm'] and r['model_seed']==row['model_seed']))
def config(row):
 if row not in rows():raise ValueError('Unregistered frozen source model')
 return dict(schema='source_feature_ce_gradient_v1',run_id=RUN,**row,source_config=source_config(row),roles=['L_s'],count=6300,batch=256,shuffle_seed=2026101501,strong_dropout_seed=2026101502,strong_noise_std=.015,feature_coefficient=.1,label_smoothing=.01,groups=list(GROUPS),parameters=list(PARAMETERS),feature='actual joint160 ownEMAweak/studentstrong',gradient_definition='Same physical L strong graph: trueCE smoothing.01 vs original feature consistency; four teacher/student correctness groups share full batch denominator; no training or fitted head',student_mode='train_no_BN_state_update',teacher_mode='eval_stopgrad',source_only=True,query_read=False,U_IQ_read=False,U_truth_read=False,V_forward=False,head_updates=0,optimizer_steps=0,model_state_updates=0,extra_training_epochs=0,checkpoint_epoch=50,output_root=(BASE/row['row_id']).as_posix(),launch_owner=OWNER)
def validate(c):
 row=next((r for r in rows() if r['row_id']==c.get('row_id')),None)
 if row is None or c!=config(row):raise ValueError('Frozen source action differs')
def model_api(row):
 if row['model_seed']==SEEDS[0]:from experiments.cvs_feature_mean_teacher50 import predict
 else:from experiments.cvs_feature_mean_teacher_seeds import predict
 return predict
