"""One fixed source identity-preservation question, before E20 Ag training."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
PROJECT='/home/szu2070436088/2510044040/CV-SincNet'
RUN='20261016-diagnostic-F-crossRX-sameTX-IQmix-fitL-manysig-m2-r01'
RELEASE='cvs_F_sameTX_IQmix_fitL_20261016_r01'
OWNER='codex/root/F-sameTX-IQmix-source-20261016'
LOCAL='local_artifacts/F_sameTX_IQmix_source_20261016'
BASE=Path(PROJECT)/'runs'/RUN
SOURCE=PROJECT+'/runs/phase1_daot_rc4_pure_game_m3_20260917_r2/source_contract.json'
PREPARATION=PROJECT+'/runs/20261011-phase1-stage2-geometry-source-manysig-m3-r01/s2026092701'
FIT=(3,4,6,8);SEEDS=(2026092701,2026092702);LAMBDA=.1
MODELS={
2026092701:dict(run='20261016-phase1-F-crossRX-satmix25-sourceRX1-e20-s2026092701-m2-r01',backend=PROJECT+'/releases/cvs_F_crossRX_satmix25_source_e20_20261016_r01',commit='0ae066adb10a001e834c8c679c5b7b81c20b3ac8'),
2026092702:dict(run='20261016-phase1-F-crossRX-satmix25-confirm-sourceRX1-e20-s2026092702-m2-r01',backend=PROJECT+'/releases/cvs_F_crossRX_satmix25_confirm_source_e20_20261016_r01',commit='d71531d53c1d9ec46e3915963e05bff96760e80e')}
for seed,v in MODELS.items():v['source']=PROJECT+'/runs/'+v['run']+'/B00-RX1-s'+str(seed)+'/source'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+chr(10),encoding='utf8')
def config():
 return dict(run_id=RUN,launch_owner=OWNER,output_root=BASE.as_posix(),question='Do mild same-TX cross-RX observed-IQ mixtures preserve frozen default-head identities before training?',models={str(k):v for k,v in MODELS.items()},model='Own F+CROSS_RX,3SIG_TIME,unifiedSinc,joint160,defaultcos30,NoExplicitCFO;fixedscratchE20final20;no new head',model_role='Frozen source diagnostic only,not future training initialization',source_contract=SOURCE,preparation=PREPARATION,fit_RX=list(FIT),days=[1,2,3],TX=list(range(6)),L_unique=5040,samples_per_cell=70,donor_scope='Same visibleL TX and same day,other fitRX;all3 otherRX peranchor;not simultaneous emission',pairing='Sort opaqueIDs within each TX/RX/day cell;donor next rank modulo70,all otherRX;sameRX next-rank control;no class-specific parameter',operator='Normalize donor to anchor RMS,align donor relative lag20 frequency and window80:160 commonphase to anchor,mix .9anchor+.1donor then restore anchorRMS;all256samples,no shift/crop/noise/channel/PA',mix_coefficient=LAMBDA,bound='At most 2lambda/(1-2lambda)=.25 relative RMS before float rounding;engineering bound,not a measured RX strength',NoExplicitCFO=True,internal_reference_compensation_retained=True,anchor_frequency_not_zeroed=True,normalization='Native already packetRMS;restore anchorpower after combination;cannot be eliminated as simple gain',views=dict(raw=5040,sameRX_sameTX=5040,crossRX_sameTX=15120),source_NN_per_model=25200,total_NN_packets=50402,synthetic_CP_smoke_NN=2,new_training_epochs=0,optimizer_updates=0,teacher_forwards=0,checkpoint_loads=2,old_training_cost_separate='2 scratchE20/8880updates/9373356 NN across frozen B00 models;verify prior records,no repeat',criteria=dict(min_overall_accuracy_gain_pp=-.5,min_overall_macroF1_gain_pp=-.5,min_each_RX_accuracy_gain_pp=-1.,min_each_TX_accuracy_gain_pp=-1.,min_each_day_accuracy_gain_pp=-1.,min_cross_vs_sameRX_accuracy_gain_pp=-.5,min_raw_correct_retention=.99,min_median_relative_IQ_change=.01,max_relative_IQ_change=.25001,max_relative_power_error=1e-5),on_fail='Close fixed same-TX IQmix .1 alignment recipe;no coefficient/alignment/window/RX/seed rescue',on_pass='Identity guard only;next matched scratchE20 raw-slot/sameRXmix/crossRXmix required;not Ag,DG or joint success',future_training='F+CROSS_RX/E20,baseline raw.75+ordinary.25 preserved;extra .25 raw or sameRXmix or crossRXmix,8L forwards,original hiddenU/ownEMA unchanged;separately register before training',U_used=False,U_truth_read=False,V_used=False,held_RX1_used=False,target_access=False,automatic_training=False,default_head_only=True,device='CPU',threads=2,source_basis=['https://arxiv.org/pdf/2112.15363','https://arxiv.org/abs/1710.09412'],limitations=['Same-label convex mixing does not prove fingerprint preservation','No unique RXhardware replacement/inversion;RX nonlinearity and packet variability remain','Same-RX control distinguishes some averaging/extra-view effects;only future heldRX E20 can establish DG','Known preamble data,25Msps after equalization and CFO reapplication;anchor CFO is retained but never identity input'])
def validate(c):
 if c!=config():raise ValueError('Fixed same-TX IQmix data/operator/budget/criteria differ')
