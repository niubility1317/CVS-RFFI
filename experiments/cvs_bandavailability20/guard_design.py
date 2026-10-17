"""Fixed two-source-model identity guard;not an Ag training claim."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
RUN='20261017-phase1-FcrossRX-bandavailability-sourceRX1-e20-s2026092701-m4-r01'
BASE='PROJECT/runs/'+RUN+'/guard';SOURCE='PROJECT/runs/phase1_daot_rc4_pure_game_m3_20260917_r2/source_contract.json';PREPARATION='PROJECT/runs/20261011-phase1-stage2-geometry-source-manysig-m3-r01/s2026092701'
OWNER='codex/root/FcrossRX-bandavailability-20261017';FIT=(3,4,6,8);SEEDS=(2026092701,2026092702);DRAW_SEEDS=(202610171911,202610171912,202610171913,202610171914)
MODELS={2026092701:{"run":"20261016-phase1-F-crossRX-satmix25-sourceRX1-e20-s2026092701-m2-r01","backend":"PROJECT/releases/cvs_F_crossRX_satmix25_source_e20_20261016_r01","commit":"0ae066adb10a001e834c8c679c5b7b81c20b3ac8","source":"PROJECT/runs/20261016-phase1-F-crossRX-satmix25-sourceRX1-e20-s2026092701-m2-r01/B00-RX1-s2026092701/source"},2026092702:{"run":"20261016-phase1-F-crossRX-satmix25-confirm-sourceRX1-e20-s2026092702-m2-r01","backend":"PROJECT/releases/cvs_F_crossRX_satmix25_confirm_source_e20_20261016_r01","commit":"d71531d53c1d9ec46e3915963e05bff96760e80e","source":"PROJECT/runs/20261016-phase1-F-crossRX-satmix25-confirm-sourceRX1-e20-s2026092702-m2-r01/B00-RX1-s2026092702/source"}}
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
def resolve(alias):
 if not alias.startswith('PROJECT/'):raise ValueError('Fixed project alias only')
 return Path('/home/szu2070436088/2510044040/CV-SincNet')/alias[len('PROJECT/'):]
def config():
 return dict(run_id=RUN,source_only=True,target_access=False,fit_L=5040,views_per_model=dict(raw=5040,MASK=20160,GAIN=20160),source_decisions_per_model=45360,expected_guard_NN=90720,guard_checkpoint_reads=2,parent_compatibility_CP_reads=1,parent_smoke_NN=2,draw_seeds=list(DRAW_SEEDS),operator='One symmetric Gaussian STF harmonic pair k/20,kUniform1..6;512FFT reflect128,sigma1/256,depth.5,allpacketrelativeRMS cap.1;gain control exactly matches outputRMS',criteria=dict(pooled_and_eachdraw_accuracy_min_pp=-.5,pooled_and_eachdraw_F1_min_pp=-.5,raw_correct_retention_min=.99,each_RX_TX_day_accuracy_min_pp=-1.,IQ_change_median_min=.01,IQ_change_max=.100001,boundary_error_to_change_max=.01,matched_power_error_max=1e-6),on_fail='Close fixed STF-pair availability mask,no width/depth/frequency/conditioning/RX/seed/threshold rescue;0newE20',on_pass='Only identity guard;auto fixedsource scratchE20 matrix RAW/GAIN/ORDINARY/MASK underpredeclared thresholds,not Agvalidity')
def validate(c):
 if c!=config():raise ValueError('Fixed guard only')
