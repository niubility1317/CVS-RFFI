"""Frozen matched E30 source interventions; zero training or target access."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
PROJECT=Path('/home/szu2070436088/2510044040/CV-SincNet')
RUN='20261017-diagnostic-FcrossRX-practical-attribution-e30-s2026092701-m2-r01';RELEASE='cvs_FcrossRX_practical_attribution_E30_20261017_r01';BASE=PROJECT/'runs'/RUN
PARENT=PROJECT/'runs'/'20261017-phase1-FcrossRX-cycleorder-factorial-e30-s2026092701-m4-r01'
PANEL=PROJECT/'runs'/'20261017-diagnostic-FcrossRX-fullsource-residual-s2026092701-m3-r01'/'diagnostic'
SOURCE=PROJECT/'runs/phase1_daot_rc4_pure_game_m3_20260917_r2/source_contract.json'
SCENES=('practical_low_suburban','practical_mid_urban','practical_low_urban')
CONDITIONS=('full','no_noise/fixed','no_noise/total','flat_H/fixed','flat_H/total','no_carrier/fixed','no_carrier/total','no_RX_IQ/fixed','no_RX_IQ/total','actual_power_quality','edge_boundary')
ARMS=('B00','B01');N=1800;BATCH=90;NEW_NN_PER_MODEL=108002
FP32=dict(cudnn_allow_tf32=False,cuda_matmul_allow_tf32=False,cudnn_benchmark=False,cudnn_deterministic=False,matmul_precision='highest')
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def write(p,q):Path(p).write_text(json.dumps(q,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
def validate_metadata(r,o):
 if r.get('role')!='V' or r.get('counts')!=dict(L_s=6300,U_s=56700,V=27000) or r.get('V_panel')!=N or r.get('held_RX') is not None or r.get('extra_Ldev') is not False or r.get('U_truth_used') is not False or r.get('target_access') is not False or r.get('updates')!=0:raise ValueError('Only original full-source V1800;no heldRX/Utruth/target/updates')
 if o.get('status')!='SOURCE_OBSERVATIONS_COMPLETE' or o.get('unique_V')!=N or o.get('draws')!=2 or o.get('scenes')!=list(SCENES) or o.get('conditions')!=list(CONDITIONS) or o.get('physical_draws')!=10800 or o.get('target_access') is not False or o.get('training_updates')!=0 or o.get('formal_test_changed') is not False:raise ValueError('Exact original total/fixed source observation matrix required')
def config():return dict(run_id=RUN,arms=list(ARMS),source_role='V',full_source_counts=dict(L_s=6300,U_s=56700,V=27000),source_RX=[1,3,4,6,8],source_days=[1,2,3],unique_source_V=N,channel_draws=2,scenes=list(SCENES),conditions=list(CONDITIONS),checkpoint_parent=str(PARENT),checkpoint_selection='own model FINAL30/6660,not EMA',cached_full_and_clean_reused=True,new_NN=216004,new_native_channel_calls=0,training_updates=0,head_fit_updates=0,target_access=False,formal_scores_read=False,model_seed_count=1,classification_logit_scale=30,criterion='descriptive repeatability: removal sensitivity reduction>=1pp in BOTH draws, reported separately per condition/scene;no candidate promotion or automatic training',stratification='original full-observation state/lock/added-noise-bin/RX;never postintervention strata')
