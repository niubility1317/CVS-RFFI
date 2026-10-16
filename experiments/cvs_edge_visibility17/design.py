"""Fixed source identity test for one bounded edge-visibility candidate."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
RUN='20261017-diagnostic-F-crossRX-edgevisibility-fitL-manysig-m2-r01'
RELEASE='cvs_F_edge_visibility_fitL_20261017_r01'
OWNER='codex/root/F-edge-visibility-source-20261017'
LOCAL='local_artifacts/F_edge_visibility_source_20261017'
BASE='PROJECT/runs/'+RUN
SOURCE='PROJECT/runs/phase1_daot_rc4_pure_game_m3_20260917_r2/source_contract.json'
PREPARATION='PROJECT/runs/20261011-phase1-stage2-geometry-source-manysig-m3-r01/s2026092701'
FIT=(3,4,6,8);SEEDS=(2026092701,2026092702)
MODELS={2026092701:dict(run='20261016-phase1-F-crossRX-satmix25-sourceRX1-e20-s2026092701-m2-r01',backend='PROJECT/releases/cvs_F_crossRX_satmix25_source_e20_20261016_r01',commit='0ae066adb10a001e834c8c679c5b7b81c20b3ac8'),2026092702:dict(run='20261016-phase1-F-crossRX-satmix25-confirm-sourceRX1-e20-s2026092702-m2-r01',backend='PROJECT/releases/cvs_F_crossRX_satmix25_confirm_source_e20_20261016_r01',commit='d71531d53c1d9ec46e3915963e05bff96760e80e')}
for seed,v in MODELS.items():v['source']='PROJECT/runs/'+v['run']+'/B00-RX1-s'+str(seed)+'/source'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+chr(10),encoding='utf8')
def resolve(alias):
 if not alias.startswith('PROJECT/'):raise ValueError('Registered project-relative path required')
 return Path(read(ROOT/'runtime.json')['project'])/alias[len('PROJECT/'):]
def config():
 return dict(run_id=RUN,launch_owner=OWNER,output_root=BASE,models={str(k):v for k,v in MODELS.items()},model='Frozen F+CROSS_RX,3SIG_TIME,unifiedSinc,joint160,defaultcos30,NoExplicitCFO;two scratchE20final20,0newhead',model_role='Source-fit operation guard only;not future training initialization',source_contract=SOURCE,preparation=PREPARATION,fit_RX=list(FIT),days=[1,2,3],TX=list(range(6)),L_unique=5040,samples_per_cell=70,question='Does fixed one-sided 16point visibility reduction preserve identities under both frozen default heads?',operator='One edge only:outer8zero,inner8halfcosine to1;rightmirror;same realwindow onI/Q;all other samples unchanged',edge_support=16,zero_samples=8,taper_samples=8,reference=[80,160],views=['raw','left','right'],source_NN_per_model=15120,total_NN_packets=30242,synthetic_CP_smoke_NN=2,checkpoint_loads=2,new_training_epochs=0,optimizer_updates=0,head_fit_updates=0,teacher_forwards=0,criteria=dict(min_overall_accuracy_gain_pp=-.5,min_overall_macroF1_gain_pp=-.5,min_each_RX_accuracy_gain_pp=-1.,min_each_TX_accuracy_gain_pp=-1.,min_each_day_accuracy_gain_pp=-1.,min_raw_correct_retention=.99,min_median_relative_IQ_change=.01,max_relative_IQ_change=.35,max_power_increase=1e-6),normalization='NativepacketRMS only;no postmask globalRMS/powerrestore;inputreference andsupportoutside bitwiseunchanged',old_training_cost='2scratchE20/8880updates/9373356NN;separate historical frozen model cost,no repeat',on_fail='Close fixed16support/8zero8taper;no length/edge/windowshape/threshold/RX/seed rescue',on_pass='Source-fit identityguard only;separate matchedscratchE20raw-slot vsedgeview needed forheldRXclean;no automatictraining',future_training='Common raw.75/ordinary.25/originalSupCon plus.25raw or.25edge sameIQ RDrop pair,8Lforwards/originalU/defaulthead joint;E20;separately register',future_source_criteria='clean>=+1pp;macroF1>=0;eachpractical/seenV>=-.5pp;single seed notjointproof',U_used=False,U_truth_read=False,V_used=False,held_RX1_used=False,target_access=False,automatic_training=False,default_head_only=True,NoExplicitCFO=True,internal_reference_compensation_retained=True,device='CPU',threads=2,source_basis=['https://arxiv.org/html/2601.03063','https://onlinelibrary.wiley.com/doi/full/10.1155/2024/7891968','https://arxiv.org/html/2412.08885'],limitations=['Literature ADSB masking not currentWiFi/crossRX or independentmask efficacy','Partial observation can erase identities;sourceguard is necessary evidence,not DGproof','Digital edgewindow does not replace RXhardware,shift timing or reduce actualnoise','Inputreference unchanged does not imply allinternal features/quality unchanged'])
def validate(c):
 if c!=config():raise ValueError('Fixed edgevisibility question differs')
