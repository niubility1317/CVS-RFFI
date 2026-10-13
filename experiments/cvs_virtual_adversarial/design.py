from copy import deepcopy
from pathlib import Path
from experiments.cvs_scratch30_baseline import design as parent
ROOT=Path(__file__).resolve().parents[2]
PROJECT,SOURCE,CAPSULE,TRUTH,CLASSES,SEEDS,FULL_FP32_POLICY,read,write=(getattr(parent,k) for k in ('PROJECT','SOURCE','CAPSULE','TRUTH','CLASSES','SEEDS','FULL_FP32_POLICY','read','write'))
RUN='20261013-phase1-virtual-adversarial-clean-manysig-m12-r02';RELEASE='cvs_virtual_adversarial_20261013_r02';OWNER='codex/root/virtual-adversarial-20261013-r02';BASE=Path(PROJECT)/'runs'/RUN
EPOCHS=50;STEPS_PER_EPOCH=222;UPDATES=EPOCHS*STEPS_PER_EPOCH;ACTIVE_U_UPDATES=UPDATES-1110
ARMS=('CE50','RANDOM_INPUT','VAT_INPUT','VAT_FEATURE');PRIMARY='VAT_FEATURE';VIEWS=('clean',);SCENES=parent.SCENES;VIEWS_ROOT=parent.VIEWS_ROOT;LEVELS=('time_down','t2.pool')
def rows():return [dict(row_id=f'{a}-s{s}',arm=a,model_seed=s) for s in SEEDS for a in ARMS]
def config(row):
 if row not in rows():raise ValueError('Unregistered VAT row')
 c=deepcopy(parent.config(next(r for r in parent.rows() if r['arm']=='CE30' and r['model_seed']==row['model_seed'])))
 enabled=row['arm']!='CE50'
 c['checkpoint_selection']='fixed_final_epoch50'
 c['optimizer_budget']=dict(epochs=EPOCHS,steps_per_epoch=STEPS_PER_EPOCH,updates=UPDATES,maximum_authorized_epochs=50,schedule='epoch cosine50')
 c.update(run_id=RUN,**row,epochs=EPOCHS,updates=UPDATES,output=(BASE/row['row_id']/'source').as_posix(),launch_owner=OWNER,source_U=enabled,U_cache=parent.U_CACHE.as_posix() if enabled else None,test_views=list(VIEWS),semi=dict(enabled=enabled,warmup_updates=1110,ramp_updates=1110,weight=.25,confidence_gate=False,min_streak=None,EMA=False,target='current student deterministic detached distribution; regenerated each update'),vat=dict(enabled=enabled,role='U_s only; labels never provided',space='alternate shallow/middle feature' if row['arm']=='VAT_FEATURE' else 'IQ input',levels=list(LEVELS) if row['arm']=='VAT_FEATURE' else [],direction='random same radius with discarded matched power gradient' if row['arm']=='RANDOM_INPUT' else 'one KL power iteration',xi_relative=.001,epsilon_relative=.02,power_iterations=1,norm='per packet full tensor L2; radius relative to detached clean tensor L2; not per-batch',KL='float64 logsoftmax/probability difference; model forward fullFP32',weak_and_search='eval dropout off; mode/RNG restored',regularized_forward='eval with gradients to identity params; no dropout or teacher copy',zero_gradient='deterministic original random direction fallback recorded, not technical failure',warmup='no U iteration/forward during first5epochs',identity_only=True),source_validation='original V health only at epochs10/20/30; no selection/promotion',goal=dict(primary=PRIMARY,clean_three_seed_mean=.84,feedback_to_training=False),budget_authorization='user20261013 permits chosen epoch budget<=50; fixed scratch50 for all12 rows, ordinarycos160, allclean, noVgate')
 return c
def validate(c):
 r=next((r for r in rows() if r['row_id']==c.get('row_id')),None)
 if r is None or c!=config(r):raise ValueError('Fixed VAT recipe differs')
 return c
