"""Full-source frozen-default-head residual diagnosis. Never a training launcher."""
from pathlib import Path
import json
ROOT=Path(__file__).resolve().parents[2]
PROJECT=Path('/home/szu2070436088/2510044040/CV-SincNet')
RUN='20261017-diagnostic-FcrossRX-fullsource-residual-s2026092701-m3-r01'
RELEASE='cvs_FcrossRX_fullsource_residual_diag_20261017_r01'
BASE=PROJECT/'runs'/RUN
OWNER='codex/root/fullsource-residual-diagnostic-20261017'
SOURCE=PROJECT/'runs/phase1_daot_rc4_pure_game_m3_20260917_r2/source_contract.json'
PREPARATION=PROJECT/'runs/20261011-phase1-stage2-geometry-source-manysig-m3-r01/s2026092701'
REFERENCE_RUN='20261017-phase1-FcrossRX-fullsource-concat-e20-s2026092701-m2-r01'
RAW_RUN='20261017-phase1-FcrossRX-fullsource-rawpractical-e20-s2026092701-m1-r01'
REFERENCES={n:PROJECT/'runs'/r/(n+'-s2026092701')/'source' for n,r in [('B00',REFERENCE_RUN),('B01',REFERENCE_RUN),('RAW_PRAC',RAW_RUN)]}
SCENES=('practical_low_suburban','practical_mid_urban','practical_low_urban')
CONDITIONS=('full','no_noise/fixed','no_noise/total','flat_H/fixed','flat_H/total','no_carrier/fixed','no_carrier/total','no_RX_IQ/fixed','no_RX_IQ/total','actual_power_quality','edge_boundary')
SEED=2026092701;PANEL_SEED=202610172401;CHANNEL_SEED=202610172402;RECEIVER_SEED=2026092701
RX=(1,3,4,6,8);DAYS=(1,2,3);DRAWS=(0,1);N=1800;BATCH=90;GRAD_N=90
FP32=dict(cudnn_allow_tf32=False,cuda_matmul_allow_tf32=False,cudnn_benchmark=False,cudnn_deterministic=False,matmul_precision='highest')
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def write(p,q):Path(p).write_text(json.dumps(q,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf8')
def config():
 return dict(run_id=RUN,source_role='V',source_counts=dict(L_s=6300,U_s=56700,V=27000),source_RX=list(RX),source_days=list(DAYS),panel_count=N,panel_per_TX_RX_day=20,panel_seed=PANEL_SEED,channel_seed=CHANNEL_SEED,receiver_seed=RECEIVER_SEED,draws=list(DRAWS),scenes=list(SCENES),conditions=list(CONDITIONS),checkpoints={n:str(p/'final_ssdg.pth') for n,p in REFERENCES.items()},model_seed=SEED,model_seed_count=1,extra_held_RX=None,extra_Ldev=False,training_updates=0,head_fit_updates=0,target_access=False,automatic_training=False,formal_test_changed=False,gradient_scope='eval-mode smoothedCE gradients; not actual stochastic RDrop/SupCon training gradients',expected_NN=363696)
def validate(c):
 if c!=config():raise ValueError('Only preregistered original-source-V/frozen-default-head matrix allowed')
 return c
def validate_input(meta):
 if meta.get('role')!='V' or meta.get('counts')!=dict(L_s=6300,U_s=56700,V=27000) or meta.get('held_RX') is not None or meta.get('U_truth_used') or meta.get('target_access') or meta.get('updates')!=0:raise ValueError('No held domain, U truth, target data or updates allowed')
