"""Isolated native AWGN(raw) identity guard under two frozen default heads."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
RUN='20261017-diagnostic-F-crossRX-isolatedAWGN-fitL-manysig-m2-r01'
RELEASE='cvs_F_isolated_AWGN_fitL_20261017_r01'
OWNER='codex/root/F-isolated-AWGN-source-20261017'
LOCAL='local_artifacts/F_isolated_AWGN_source_20261017'
BASE='PROJECT/runs/'+RUN
SOURCE='PROJECT/runs/phase1_daot_rc4_pure_game_m3_20260917_r2/source_contract.json'
PREPARATION='PROJECT/runs/20261011-phase1-stage2-geometry-source-manysig-m3-r01/s2026092701'
FIT=(3,4,6,8);SEEDS=(2026092701,2026092702);AUG_SEEDS=(202610172101,202610172102,202610172103,202610172104)
MODELS={2026092701:dict(run='20261016-phase1-F-crossRX-satmix25-sourceRX1-e20-s2026092701-m2-r01',backend='PROJECT/releases/cvs_F_crossRX_satmix25_source_e20_20261016_r01',commit='0ae066adb10a001e834c8c679c5b7b81c20b3ac8'),2026092702:dict(run='20261016-phase1-F-crossRX-satmix25-confirm-sourceRX1-e20-s2026092702-m2-r01',backend='PROJECT/releases/cvs_F_crossRX_satmix25_confirm_source_e20_20261016_r01',commit='d71531d53c1d9ec46e3915963e05bff96760e80e')}
for seed,v in MODELS.items():v['source']='PROJECT/runs/'+v['run']+'/B00-RX1-s'+str(seed)+'/source'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
def resolve(alias):
 if not alias.startswith('PROJECT/'):raise ValueError('Registered project alias required')
 return Path(read(ROOT/'runtime.json')['project'])/alias[len('PROJECT/'):]
SNR=(20.,36.)
def config():
 return dict(run_id=RUN,launch_owner=OWNER,output_root=BASE,models={str(k):v for k,v in MODELS.items()},baseline='F+CROSS_RX/3SIGTIME/unifiedSinc/joint160/defaultcos30/NoExplicitCFO',source_contract=SOURCE,preparation=PREPARATION,claim='Isolatednoise sourceidentity guard only,not cleanDG/As/trainingefficacy',question='Does nativeisolated AWGN(raw),existing20to36dB,retain allsource identities without otherordinarychanges?',fit_RX=list(FIT),days=[1,2,3],TX=list(range(6)),L_unique=5040,cells=72,samples_per_cell=70,augmentation_seeds=list(AUG_SEEDS),shuffle_seeds=[202610172110+i for i in range(4)],operator='Originalnative propercomplexAWGN(raw),SNR uniform20..36 perpacket usingrawpower;no multipath/PA/phase/time/RMS/clip/redraw',nominal_SNR_dB=list(SNR),nominal_not_total_recording_SNR=True,Gaussian_amplitude_strictly_bounded=False,retain_all_tail_samples=True,views_per_model=dict(raw=5040,noise=20160),source_NN_per_model=25200,total_NN_packets=50402,synthetic_CP_smoke_NN=2,source_noise_generations=40320,synthetic_native_canonical_generations=128,checkpoint_loads=2,new_training_epochs=0,optimizer_updates=0,teacher_forwards=0,head_fit_updates=0,criteria=dict(min_accuracy_gain_pp=-.5,min_macroF1_gain_pp=-.5,min_raw_correct_retention=.99,min_each_group_accuracy_gain_pp=-1.,require_each_draw_same_guard=True,min_IQ_change_median=.01,max_IQ_change=.15),on_fail='ClosefixedisolatedAWGN20..36/fullcoverage,no raisingSNR/clip/taildelete/redraw/conditioning/draw/RX/seed rescue',on_pass='Sourceidentity only;next three matchedscratchE20 rawslot/ordinaryslot/noiseslot,commonoldordinary unchanged,not automatictraininginthislauncher',future_E20='Commonraw.75/ordinary.25/SupCon/originalU unchanged,extra.25 raw/sameordinary/noise(raw),8Lforwards/weight1.25/sameIQ RDrop,seed01noise mustbeatbothcontrols clean>=+1pp,F1>=0,eachpractical/seenV>=-.5pp before independentconfirm',NoExplicitCFO=True,internal_compensation_retained=True,default_head_only=True,source_only=True,target_access=False,U_used=False,V_used=False,held_RX1_used=False,automatic_training=False,device='CPU',threads=2,historical_model_cost=dict(model_epochs=40,updates=8880,NN=9373356),source_basis=['analysis/cvs_isolated_awgn_candidate_20261017.md','automation_reports/CV-SincNet/20261017-diagnostic-F-crossRX-ordinaryChannel-fitL-manysig-m2-r01/report.md'])
def validate(c):
 if c!=config():raise ValueError('Fixed isolatedAWGN source question differs')
