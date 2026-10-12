from copy import deepcopy
from pathlib import Path
from experiments.cvs_scratch30_baseline import design as b
ROOT=Path(__file__).resolve().parents[2]
PROJECT,SOURCE,CAPSULE,TRUTH,CLASSES,SEEDS,FULL_FP32_POLICY,read,write=(getattr(b,k) for k in ('PROJECT','SOURCE','CAPSULE','TRUTH','CLASSES','SEEDS','FULL_FP32_POLICY','read','write'))
RUN='20261013-phase1-feedback-teacher-clean-manysig-m12-r01';RELEASE='cvs_feedback_teacher_clean_20261013_r01';OWNER='codex/root/feedback-teacher-clean-20261013';BASE=Path(PROJECT)/'runs'/RUN;ARMS=('TEACHER_SSL','FEEDBACK_TEACHER');COUNT=168000;VIEWS=('clean',)
def rows():return [dict(row_id=f'{a}-s{s}',arm=a,model_seed=s) for a in ARMS for s in SEEDS]
def readouts(c):return ('STUDENT','TEACHER')
def config(row):
 if row not in rows():raise ValueError('Unregistered feedbackteacher row')
 c=deepcopy(b.config(next(r for r in b.rows() if r['arm']=='CE30' and r['model_seed']==row['model_seed'])))
 c.update(run_id=RUN,**row,output=(BASE/row['row_id']/'source').as_posix(),launch_owner=OWNER,source_U=True,U_cache=b.U_CACHE.as_posix(),test_views=['clean'],readouts=['STUDENT','TEACHER'],source_validation='noVforward/performancegate;fixed30clean directly',
  classifier=dict(kind='single ordinarycos inbothmodels',scale=30.,dimension=160,RX_heads=False,routing=False,ensemble=False,Phase2='studentonly;discardclassweight/refitlegalsupport;teacher discarded'),
  feedback_teacher=dict(teacher_init_seed=row['model_seed']+10000,student_init_seed=row['model_seed'],models_per_row=2,epochs_each=30,updates_each=6660,mainL='all5RX26/26/26/25/25 RX/day/TX balanced same128L',feedback_RX='sorted5fitRX[(step-1)%5]',feedbackL='subset same128L feedbackRX25/26;noextraL/V',U='128 onlyother4fitRX32each',gradient_parameters='student66 optimizerbackbone excludingcosweight;teacheraux66 backbone excludingcosweight',P='frozencurrentstudent AdamW biascorrectedsecondmoment diagonal',virtual_direction='P*gM normalized by Euclidean fullbackbone norm',finite_difference_epsilon=.001,reward='logp(theta+eps*d)-logp(theta-eps*d) /2eps,centerall6classes;detach batchRMSfloor1e-4,clip2',teacher_aux='mean sum(qteacher*reward);positive sign minimizes firstorder studentfeedbackrisk proxy;teacherbackbone only',teacher_aux_weight=.1,student_KL_weight=.25,warmup_updates=1110,ramp_updates=1110,confidence=.95,streak=2,noise_std=.015,EMA=False,control='same2models/L/U/virtualforward/gradientdiagnostics;teacher onlyCE,auxweight0',method_claim='MPL-inspired frozenP firstorder finite-difference reward,not originalMPL/exactvirtualAdamW',restore='exact copy allvirtualstudentparams before actualstudentforward;noqueryvirtualfit'),
  goal=dict(primary='FEEDBACK_TEACHER student',clean_three_seed_mean=.84,feedback_to_training=False),budget_authorization='user30eachmodel/commoncos/noVperformancegate/clean;earlierreport fixedteacher-target feedback mechanism;6rows12freshmodels79920combinedupdates')
 return c
def validate(c):
 row=next((r for r in rows() if r['row_id']==c.get('row_id')),None)
 if row is None or c!=config(row):raise ValueError('Fixed feedbackteacher matrix differs')
 return c
