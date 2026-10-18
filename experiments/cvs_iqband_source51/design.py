import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
PROJECT='/home/szu2070436088/2510044040/CV-SincNet'
RUN='20261018-diagnostic-FcrossRX-iqband-LtoV-manysig-m1-r01'
BASE=Path(PROJECT)/'runs'/RUN
OBS=Path(PROJECT)/'runs'/'20261018-diagnostic-FcrossRX-iqskew-LtoV-manysig-m1-r01'/'observations.npz'
INPUT_CONFIG=OBS.with_name('resolved_config.json')
RX=(1,3,4,6,8)
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
def config():
 return dict(run_id=RUN,source_only=True,input_statistics=str(OBS),input_config=str(INPUT_CONFIG),source_RX=list(RX),source_days=[1,2,3],TX=list(range(6)),L=6300,V=27000,U_hidden=56700,U_used=False,IQ_shape=[2,256],fs_Hz=25000000,source_RX_holdout=None,positive_rfft_bins=list(range(1,128)),observable='10log10(cell mean I power/cell mean Q power),then frequency mean removed',basis='(k/127)^2 then frequency mean removed',edge_to_zero_band_differential_gain_cap_dB=1.0,coefficient_grid_points=101,reference_RX=1,folds='18 heldTX/day;fit other5TX other2day;all5RX retained',V_role='evaluation only;no V parameter fit or perpacket pairing',L_split='nonoverlapping deterministic parity halves within each70packet cell',criterion=dict(identity_SSE_min=1e-12,mean_split_RX_contrast_power_min_dB2=.01,each_RX_split_power_positive=True,shared_identity_ratio_max=.8,shared_additive_ratio_max=.9,folds_improving_additive_min=15,each_RX_no_worse_additive=True,shared_TXconditional_ratio_max=1.05,boundary_fits_max=0),NN=0,checkpoint_reads=0,IQ_reads=0,training_updates=0,head_fits=0,new_IQ_views=0,target_access=False,claim='Necessary source differential I/Q frequency gain law only;not identified receiver hardware,finitewindow identity or Ag gain')
