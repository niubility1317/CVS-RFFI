"""Fixed L-only paired noise-gradient falsification;no new training or target access."""
import json
from pathlib import Path
PROJECT=Path('/home/szu2070436088/2510044040/CV-SincNet')
ROOT=Path(__file__).resolve().parents[2]
RUN='20261018-diagnostic-FcrossRX-noise-headgradient-sourceL-s2026092701-m1-r01'
BASE=PROJECT/'runs'/RUN
PARENT=PROJECT/'runs/20261018-phase1-FcrossRX-cleanurban-U-e30-s2026092701-m3-r01'
BACKEND=PROJECT/'releases/cvs_FcrossRX_cleanurban_U_E30_20261018_r01'
PHYSICS=PROJECT/'releases/cvs_FcrossRX_fullsource_residual_diag_20261017_r01'
SOURCE=PROJECT/'runs/phase1_daot_rc4_pure_game_m3_20260917_r2/source_contract.json'
CACHE=PROJECT/'runs/20261011-phase1-stage2-geometry-source-manysig-m3-r01/s2026092701'
N=900;BATCH=30;RX=(1,3,4,6,8);DAYS=(1,2,3)
SCENES=('practical_mid_urban','practical_low_urban')
CONDITIONS=('full','no_noise/fixed','no_noise/total')
PANEL_SEED=20261018461;CHANNEL_SEED=20261018462;HARDWARE_SEED=20261016423
EXPECTED_NN=11702;EXPECTED_AUTOGRAD=1173
OWNER='codex/root/noise-gradient40'
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def write(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
def config():return dict(run_id=RUN,source_only=True,source_role='L_s',source_L_available=6300,panel=N,source_RX=list(RX),days=list(DAYS),TX_count=6,model_seed=2026092701,split_seed=392005,panel_seed=PANEL_SEED,channel_seed=CHANNEL_SEED,receiver_seed=HARDWARE_SEED,scenes=list(SCENES),conditions=list(CONDITIONS),draws=2,objectives=['margin','CE0','CE01'],primary_objective='margin',primary_parameter='head_W',checkpoint=str(PARENT/'B0-s2026092701/source/final_ssdg.pth'),checkpoint_selection='preregistered B0/fixed own20+10;not a formal-score winner',parent_epochs=30,parent_updates=6660,NN=EXPECTED_NN,autograd_calls=EXPECTED_AUTOGRAD,epochs=0,optimizer_updates=0,head_fit_updates=0,U_truth_read=False,V_forward=False,V_backprop=False,target_access=False,formal_scores_read=False,automatic_training=False,goal_complete=False)
