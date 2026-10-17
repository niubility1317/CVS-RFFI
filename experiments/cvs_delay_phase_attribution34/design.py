"""Four flat-kernel delay/phase controls on existing urban source latents."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
PROJECT=Path('/home/szu2070436088/2510044040/CV-SincNet');RUN='20261017-diagnostic-FcrossRX-delay-phase-e30-s2026092701-m2-r01';RELEASE='cvs_FcrossRX_delay_phase_E30_20261017_r01';BASE=PROJECT/'runs'/RUN
PARENT=PROJECT/'runs'/'20261017-phase1-FcrossRX-cycleorder-factorial-e30-s2026092701-m4-r01'
ATTRIBUTION=PROJECT/'runs'/'20261017-diagnostic-FcrossRX-practical-attribution-e30-s2026092701-m2-r01'
PANEL=PROJECT/'runs/20261017-diagnostic-FcrossRX-fullsource-residual-s2026092701-m3-r01/diagnostic'
SOURCE=PROJECT/'runs/phase1_daot_rc4_pure_game_m3_20260917_r2/source_contract.json'
SCENES=('practical_mid_urban','practical_low_urban');CACHED_SCENES=('practical_low_suburban',)+SCENES
MODES=('D00_delay24_originalphase','D10_delay0_originalphase','D01_delay24_zerophase','D11_delay0_zerophase')
GRID=((24,False),(0,False),(24,True),(0,True));ARMS=('B00','B01');N=1800;BATCH=90;NN_PER_MODEL=21602
FP32=dict(cudnn_allow_tf32=False,cuda_matmul_allow_tf32=False,cudnn_benchmark=False,cudnn_deterministic=False,matmul_precision='highest')
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def write(p,q):Path(p).write_text(json.dumps(q,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf8')
def config():return dict(run_id=RUN,source_role='V',counts=dict(L_s=6300,U_s=56700,V=27000),held_RX=None,unique_source_V=N,scenes=list(SCENES),draws=2,modes=list(MODES),model_seed=2026092701,source_panel_seed=202610172401,source_channel_seed=202610172402,receiver_seed=2026092701,target_access=False,formal_scores_read=False,training_updates=0,head_fit_updates=0,new_NN=43204,controlled_renderer_calls=28800,new_native_residual_calls=0,formal_test_changed=False,criterion='Separately delay or phase:all4 urban/draw cells B00 gain>=5pp AND original As advantage reduced by>=max(5pp,50% original advantage);report joint separately;no automatic training/seed extension',scope='flat-kernel counterfactual family,not noncausal real-waveform deconvolution;G1/noise/receiver/AGC fixed')
def validate(c):
 if c!=config():raise ValueError('Only fixed sourceV/frozen-defaulthead/zero-update/zero-target matrix allowed')
 return c
