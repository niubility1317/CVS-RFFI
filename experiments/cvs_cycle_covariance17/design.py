"""One fixed source-only cycle-residual association question;no model or augmentation."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
PROJECT='PROJECT'
RUN='20261017-diagnostic-F-crossRX-cyclecov-fitL-manysig-m1-r01'
RELEASE='cvs_F_cycle_covariance_fitL_20261017_r01'
OWNER='codex/root/F-cycle-covariance-source-20261017'
SOURCE=PROJECT+'/runs/phase1_daot_rc4_pure_game_m3_20260917_r2/source_contract.json'
PREPARATION=PROJECT+'/runs/20261011-phase1-stage2-geometry-source-manysig-m3-r01/s2026092701'
BASE=PROJECT+'/runs/'+RUN
FIT=(3,4,6,8)
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+chr(10),encoding='utf8')
def config():
 return dict(run_id=RUN,launch_owner=OWNER,output_root=BASE,source_contract=SOURCE,preparation=PREPARATION,fit_L=5040,fit_RX=list(FIT),days=[1,2,3],TX=list(range(6)),samples_per_cell=70,window=[20,180],cycle=20,cycles=8,lags=[1,2,3,4],observation='Analysis-only lag20 phase cancellation,eightcycle mean removal,within-cycle normalized proper complex lag1to4 correlations;8 real shape coordinates;power descriptive only',crossfit='18folds exclude heldTX and heldday jointly;percell medians;RX centers from other5TX/2days;center4RX in train and test;MSE vszeroRXcontrast and3fixedwrongRXrotations;no fitted classifier',criteria=dict(min_cell_median_coherence=.9,min_cell_q10_coherence=.8,min_shape_contrast_baseline_RMS=.02,min_pooled_shape_risk_gain=.2,min_positive_folds=15,min_each_RX_risk_gain=0.,min_gain_over_best_wrong_RX=.02),on_fail='Close fixed source window/cycle/lags/thresholds;no window/lag/RX/seed rescue or automatictraining',on_pass='Source RX-associated correlation rationale only;separate boundedIQ identitypreservation and matchedE20 defaulthead recognition still required',source_only=True,U_statistically_used=False,U_truth_read=False,V_statistically_used=False,held_RX1_statistically_used=False,target_access=False,IQ_modified=False,model_constructed=False,checkpoint_loads=0,network_forwards=0,optimizer_steps=0,teacher_forwards=0,head_fit_updates=0,automatic_training=False,device='CPU',threads=2,NoExplicitCFO_preserved=True,phase_estimate_not_identity_feature=True,model_boundary='Future F+CROSS_RX/3SIGTIME/unifiedSinc/joint160/defaultcos/E20/originalU unchanged;no model used here',source_basis=['https://arxiv.org/pdf/2112.15363','https://www.mathworks.com/help/wlan/gs/non-ht-ppdu-structure.html','https://arxiv.org/html/2207.03001'],limitations=['Perioddifference residual may contain TX stochastic errors,channel/equalization/resampling artifacts;not pureRXnoise','RX association is not receiverhardware identification,IQ identitypreservation,DG,As or joint proof','Carrier source cache otherroles/features deserialized but not statistically used;only originalL fields extracted'])
def validate(c):
 if c!=config():raise ValueError('Fixed cycle covariance source question differs')

def resolve(alias,runtime):
 if not alias.startswith('PROJECT/'):raise ValueError('Expected registered project-relative path')
 return Path(runtime['project'])/alias[len('PROJECT/'):]
