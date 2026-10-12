"""Six engineering hypotheses fixed before the current conditional batch scored."""
from copy import deepcopy
import math
from pathlib import Path
from experiments.cvs_scratch_boundary_style import design as old
ROOT=Path(__file__).resolve().parents[2]
PROJECT,SOURCE,CAPSULE,TRUTH,CLASSES,SEEDS,FULL_FP32_POLICY=(getattr(old,k) for k in ('PROJECT','SOURCE','CAPSULE','TRUTH','CLASSES','SEEDS','FULL_FP32_POLICY'))
read,write,baseline=old.read,old.write,old.baseline
RUN='20261012-phase1-scratch-engineering-manysig-m18-r01'
RELEASE='cvs_scratch_engineering_20261012_r01';OWNER='codex/root/scratch-engineering-20261012'
BASE=Path(PROJECT)/'runs'/RUN
ARMS=('SSL_LR3','SSL_WARM_LR3','SSL_SAM','SSL_SMOOTH','SSL_HIDDEN_MIX','SSL_WARM_LR3_SAM')
PRIMARY='SSL_WARM_LR3_SAM'
PAIRS=[(a,'P0') for a in ARMS]+[('SSL_WARM_LR3','SSL_LR3'),('SSL_WARM_LR3_SAM','SSL_WARM_LR3'),('SSL_WARM_LR3_SAM','SSL_SAM')]
REFERENCE_RUN=old.RUN
REFERENCE_PAIRS=[(a,'L_REPLAY' if a=='SSL_HIDDEN_MIX' else 'SSL') for a in ARMS]

def rows():return [dict(arm=a,model_seed=s,row_id=f'{a}-s{s}') for s in SEEDS for a in ARMS]

def config(row):
    if row not in rows():raise ValueError('Unregistered engineering row')
    prior=next(r for r in old.rows() if r['arm']=='SSL' and r['model_seed']==row['model_seed'])
    c=deepcopy(old.config(prior));arm=row['arm']
    sam=arm in ('SSL_SAM','SSL_WARM_LR3_SAM');warm=arm in ('SSL_WARM_LR3','SSL_WARM_LR3_SAM');high=arm in ('SSL_LR3','SSL_WARM_LR3','SSL_WARM_LR3_SAM')
    c.update(schema='scratch_engineering_v1',run_id=RUN,**row,method='cvs_scratch_engineering',stage='phase1_scratch_engineering',
        output_root=(BASE/row['row_id']/'source').as_posix(),launch_owner=OWNER,label_smoothing=.05 if arm=='SSL_SMOOTH' else 0.)
    c['teacher']['cache_root']=(BASE/'peer_cache').as_posix();c['evaluation']['primary_arm']=PRIMARY
    c['optimizer'].update(peak_lr=.0006 if high else .0002,schedule='step warm444 then cosine3996' if warm else 'epoch cosine20; random student20; historical peer teacher cost separate')
    c['multilevel'].update(enabled=False,statistic='none',mixstyle_l=False,mixstyle_u=False,fishr_mode='none',fishr_levels=[],head_warmup_steps=0,
        style_selection='none',aux_ce_to_backbone=False,aux_ce_routing='no auxiliary heads or CE',routing_control='engineering only; no auxiliary forward/calibration',
        fishr_scope='disabled; no Fishr graph',gradient_budget_scope='not applicable',head_warmup_batch=60)
    c['multilevel']['joint_gradient_cap']=dict(enabled=False,rho=.2,eps=1e-12)
    c['engineering']=dict(sam=sam,sam_rho=.02,sam_adaptive=False,sam_all_trainable_identity=True,
        same_batch_same_RNG=True,warm_lr=warm,warm_updates=444 if warm else 0,hidden_mix=arm=='SSL_HIDDEN_MIX',
        mix_levels=['time_down','t2.pool'],mix_layer_policy='alternate from active step445; exactly one layer per update',
        mix_beta_alpha=2.,mix_lambda_policy='max(lambda,1-lambda)',mix_source='L_s only; different TX donors; repeated donors allowed',
        mix_weight=1.,mix_inference=False,labels_smoothing_scope='L identity CE only; fixed peer U targets unchanged')
    c['enabled']={key:False for key in c.get('enabled',{})};c['enabled']['averaging']=False
    return c

def validate(c):
    row=next((r for r in rows() if r['row_id']==c.get('row_id')),None)
    if row is None or c!=config(row):raise ValueError('Engineering fixed config differs')
    return c

def learning_rate(c,epoch):
    if type(epoch) is not int or not 1<=epoch<=20:raise ValueError('Epoch outside fixed20')
    return c['optimizer']['min_lr']+(c['optimizer']['peak_lr']-c['optimizer']['min_lr'])*.5*(1+math.cos(math.pi*(epoch-1)/19))

def learning_rate_step(c,step):
    if type(step) is not int or not 1<=step<=4440:raise ValueError('Step outside fixed4440')
    if not c['engineering']['warm_lr']:return learning_rate(c,(step-1)//222+1)
    lo,hi=c['optimizer']['min_lr'],c['optimizer']['peak_lr']
    if step<=444:return lo+(hi-lo)*(step-1)/443
    return lo+(hi-lo)*.5*(1+math.cos(math.pi*(step-445)/3995))
