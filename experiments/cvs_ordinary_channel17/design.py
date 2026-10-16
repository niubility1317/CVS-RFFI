"""Source-only total-effect factorial for existing ordinary AWGN/multipath."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
RUN='20261017-diagnostic-F-crossRX-ordinaryChannel-fitL-manysig-m2-r01'
RELEASE='cvs_F_ordinary_channel_fitL_20261017_r01'
OWNER='codex/root/F-ordinary-channel-source-20261017'
LOCAL='local_artifacts/F_ordinary_channel_source_20261017'
BASE='PROJECT/runs/'+RUN
SOURCE='PROJECT/runs/phase1_daot_rc4_pure_game_m3_20260917_r2/source_contract.json'
PREPARATION='PROJECT/runs/20261011-phase1-stage2-geometry-source-manysig-m3-r01/s2026092701'
FIT=(3,4,6,8);SEEDS=(2026092701,2026092702);ARMS=('full','no_AWGN','no_MP','no_both');AUG_SEEDS=(202610171701,202610171702,202610171703,202610171704)
MODELS={2026092701:dict(run='20261016-phase1-F-crossRX-satmix25-sourceRX1-e20-s2026092701-m2-r01',backend='PROJECT/releases/cvs_F_crossRX_satmix25_source_e20_20261016_r01',commit='0ae066adb10a001e834c8c679c5b7b81c20b3ac8'),2026092702:dict(run='20261016-phase1-F-crossRX-satmix25-confirm-sourceRX1-e20-s2026092702-m2-r01',backend='PROJECT/releases/cvs_F_crossRX_satmix25_confirm_source_e20_20261016_r01',commit='d71531d53c1d9ec46e3915963e05bff96760e80e')}
for seed,v in MODELS.items():v['source']='PROJECT/runs/'+v['run']+'/B00-RX1-s'+str(seed)+'/source'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
def resolve(alias):
 if not alias.startswith('PROJECT/'):raise ValueError('Registered project alias required')
 return Path(read(ROOT/'runtime.json')['project'])/alias[len('PROJECT/'):]
def config():
 return dict(run_id=RUN,launch_owner=OWNER,output_root=BASE,models={str(k):v for k,v in MODELS.items()},baseline='F+CROSS_RX/3SIGTIME/unifiedSinc/joint160/defaultcos30/NoExplicitCFO',question='Does fixed joint removal of existingordinary AWGN/multipath repair sourceidentities without harming raw retention?',claim='Oldordinary causal audit only,not newAg orDG/As efficacy',source_contract=SOURCE,preparation=PREPARATION,fit_RX=list(FIT),days=[1,2,3],TX=list(range(6)),L_unique=5040,cells=72,samples_per_cell=70,effective_epoch=13,augmentation_seeds=list(AUG_SEEDS),shuffle_seeds=[202610171710+i for i in range(4)],arms=list(ARMS),primary_arm='no_both',diagnostic_arms=['no_AWGN','no_MP'],operator='All originalnativeops/params/PA remain;eacharm restores samebatch startingRNG;wrappers execute originalprimitive toconsume draws thenreturn input forremovedprimitive;no RMS restore',dependency='MP thenpower-relativeAWGN thenPA;totalchain effects withAWGNsigmarecomputed andPAapplied tochangedIQ;not fixedaddednoise controlleddirecteffect',As_input='ExistingAs usesraw,notordinary;no sameview duplicateAWGN/MP established;no As generated here',matching='Exactnativecallback params/Python+Torch pre/postRNG/PAstrength matched;kernel params replayed withprivategenerator forlogging only;wrappers restored',views_per_model=dict(raw=5040,full=20160,no_AWGN=20160,no_MP=20160,no_both=20160),source_NN_per_model=85680,total_NN_packets=171362,synthetic_CP_smoke_NN=2,ordinary_generations=161280,checkpoint_loads=2,new_training_epochs=0,optimizer_updates=0,teacher_forwards=0,head_fit_updates=0,criteria=dict(min_primary_vs_full_accuracy_gain_pp=.2,min_primary_vs_full_macroF1_gain_pp=0.,min_positive_draws=3,min_each_draw_gain_pp=-.2,min_primary_vs_raw_accuracy_gain_pp=-.5,min_primary_vs_raw_macroF1_gain_pp=-.5,min_each_group_vs_raw_gain_pp=-1.,min_each_group_vs_full_gain_pp=-.5,min_raw_correct_retention=.99),on_fail='Closefixed jointordinary AWGN/MP removal;do not promote singlebypass diagnostics/conditiononPA/coefficient/gainclip/draw/RX/seed rescue;not RMS revival',on_pass='Sourceidentity repair only;separate samebudget/sameforward scratchE20 required,not automatictraining',NoExplicitCFO=True,internal_compensation_retained=True,default_head_only=True,source_only=True,target_access=False,U_used=False,V_used=False,held_RX1_used=False,automatic_training=False,device='CPU',threads=2,historical_model_cost=dict(model_epochs=40,updates=8880,NN=9373356),source_basis=['analysis/cvs_joint_ordinary_IQ_audit_20261016.md','analysis/cvs_ordinary_PA_schedule_correction_20261016.md','automation_reports/CV-SincNet/20261017-diagnostic-F-crossRX-ordinaryRMS-fitL-manysig-m2-r02/report.md'])
def validate(c):
 if c!=config():raise ValueError('Fixed ordinarychannel source question differs')
