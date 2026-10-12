from copy import deepcopy
from pathlib import Path
from experiments.cvs_quality_film import design as q
from experiments.cvs_scratch30_baseline import design as b
ROOT=Path(__file__).resolve().parents[2]
PROJECT,SOURCE,FULL_FP32_POLICY,read,write,SEED,RX=(getattr(q,k) for k in ('PROJECT','SOURCE','FULL_FP32_POLICY','read','write','SEED','RX'))
RUN='20261013-phase1-cross-rx-feedback-source-manysig-s2026092701-r01';RELEASE='cvs_cross_rx_feedback_20261013_r01';OWNER='codex/root/cross-rx-feedback-20261013';BASE=Path(PROJECT)/'runs'/RUN;ARMS=('CE_MATCHED','UNIFORM_SSL','META_SSL')
def rows():return [dict(row_id=f'{a}-RX{rx}-s{SEED}',arm=a,held_rx=rx,model_seed=SEED) for rx in RX for a in ARMS]
def config(row):
 if row not in rows():raise ValueError('Unregistered crossRX feedback row')
 c=deepcopy(q.config(next(r for r in q.rows() if r['arm']=='CE30' and r['held_rx']==row['held_rx'])));c.pop('quality_film')
 c.update(run_id=RUN,**row,output=(BASE/row['row_id']).as_posix(),launch_owner=OWNER,source_U=row['arm']!='CE_MATCHED',U_cache=b.U_CACHE.as_posix() if row['arm']!='CE_MATCHED' else None,sampling='eachfitRX32 L, TX/day roundrobin completepermutation;sourceRX Uday roundrobin completepermutation, opaquePIDpassbank;sourceUtruthclosed',
  classifier=dict(kind='single ordinarycos',scale=30.,dimension=160,RX_heads=False,routing=False,ensemble=False,Phase2='discard classweight, support-onlyrefit'),
  feedback=dict(inner_RX=3,feedback_RX=1,rotation='sortedfitRX[(step-1)%4]',L_total_per_step=128,L_per_RX=32,U_total_per_step=128,U_inner_split=[43,43,42],U_microgroups=[32]*4,feedback_uses='trainingL only, not V',supervised='single CE over all128 L includes32 feedback and96 inner',gradient_parameters='all optimizer identityparameters except ordinarycos classweight;actualnames recorded',preconditioner='detached inverse sqrt biascorrected currentAdamW secondmoment plus optimizer epsilon;zero-state identity',weight_rule='positive Pmetric cosine between feedbackL gradient and each32occurrence U gradient, divideby meanpositive,cap2;allnonpositive -> zeroU',gradient='firstorder frozenP localapprox;not exact virtualAdamW/momentum/weightdecay attribution',weights_detached=True,weight=.25,confidence=.95,streak=2,warmup_updates=1110,ramp_updates=1110,strong_noise=.015,EMA=False,uniform_control='sameL/U/forward/gradient/Pdiagnostics;four U weights1',CE_control='same128L/RXrotate sampler, no Uteacher/iterator/metagrad') ,
  scientific_rule=dict(mean_gain_vs_CE=.01,mean_gain_vs_UNIFORM=.005,nonnegative_RX_each=4,worst_RX_noninferiority_each=.01),goal=dict(primary='META_SSL',scope='source_only_not84',feedback_to_training=False),full_source_recipe=dict(seeds=[2026092701,2026092702,2026092703],arms=list(ARMS),models=9,epochs_per_model=30,L_total=128,L_per_sorted_RX=[26,26,26,25,25],feedback_RX='sortedsourceRX[(step-1)%5],size25or26',U_inner_RX=4,U_per_inner_RX=[32]*4,feedback='same128 L CE, feedbackRX separate eval gradient;no Vupdate',singlecos=True,sevenviews=True,truthlast=True),budget_authorization='user scratch30 and ordinarycos; report crossRX Ufeedback route;15models99900updates')
 return c
def validate(c):
 row=next((r for r in rows() if r['row_id']==c.get('row_id')),None)
 if row is None or c!=config(row):raise ValueError('Fixed crossRX feedback matrix differs')
 return c
