"""Relative I/Q timing source premise;no IQ augmentation or model training."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
PROJECT='/home/szu2070436088/2510044040/CV-SincNet'
RUN='20261018-diagnostic-FcrossRX-iqskew-LtoV-manysig-m1-r01'
RELEASE='cvs_FcrossRX_iqskew_source_20261018_r01'
BASE=Path(PROJECT)/'runs'/RUN
CACHE=Path(PROJECT)/'runs/20261011-phase1-stage2-geometry-source-manysig-m3-r01/s2026092701/source_cache.pt'
SOURCE=Path(PROJECT)/'runs/phase1_daot_rc4_pure_game_m3_20260917_r2/source_contract.json'
RX=(1,3,4,6,8)
OWNER='codex/root/iqskew-source50'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
def config():
 return dict(run_id=RUN,source_cache=str(CACHE),source_contract=str(SOURCE),source_RX=list(RX),source_days=[1,2,3],TX=list(range(6)),L=6300,V=27000,U_hidden=56700,U_used=False,IQ_shape=[2,256],fs_Hz=25000000,source_only=True,target_access=False,model_forwards=0,training_updates=0,teacher_forwards=0,head_fits=0,new_IQ_views=0,source_RX_holdout=None,positive_rfft_bins=list(range(1,128)),DC_Nyquist_excluded=True,window='existing full256 rectangular statistic;not a new IQ observation',observable='cell cross-spectrum I*conj(Q)/sqrt(meanIpower*meanQpower)',relative_delay_samples_cap=.5,delay_grid_points=101,reference_RX=1,statistical_folds='held TX and held day;fit other5TX other2day,all5RX retained',V_role='fixed evaluation;no parameter fitting;V RX1 supplies same TX/day reference observable,not paired packets',L_identifiability='within each of90 cells split70 packets by deterministic order parity;cross-half coherence power,not raw finite-sample magnitude',criterion=dict(identity_SSE_min=1e-12,split_coherent_power_min=.0025,each_RX_split_power_positive=True,shared_identity_ratio_max=.8,shared_additive_ratio_max=.9,folds_improving_additive_min=15,each_RX_no_worse_additive=True,shared_TXconditional_ratio_max=1.05,nonreference_fits_at_boundary_max=0),claim='Necessary source cross-spectral phase-law evidence only;not identified RX hardware,valid finite-window delay,identity gain,or Ag success')
