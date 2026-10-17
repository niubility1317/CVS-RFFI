"""One fixed full-source identity/observation guard;no target or training."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
RUN='20261017-diagnostic-F-receiverBandNoise-fullL-s2026092701-m1-r01'
RELEASE='cvs_F_receiverBandNoise_source_20261017_r01'
LOCAL='local_artifacts/F_receiver_band_noise_20261017'
OWNER='codex/root/F-receiver-band-noise'
BASE='PROJECT/runs/'+RUN
SOURCE='PROJECT/runs/phase1_daot_rc4_pure_game_m3_20260917_r2/source_contract.json'
PREPARATION='PROJECT/runs/20261011-phase1-stage2-geometry-source-manysig-m3-r01/s2026092701'
REFERENCE_RUN='20261017-phase1-F-crossRX-fullsource-concat-e20-s2026092701-m3-r02'
REFERENCE_ROW='REPEAT-s2026092701'
FIT=(1,3,4,6,8)
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
def resolve(alias):
 if not alias.startswith('PROJECT/'):raise ValueError('Registered project alias required')
 return Path(read(ROOT/'runtime.json')['project'])/alias[len('PROJECT/'):]
def config():
 return dict(run_id=RUN,launch_owner=OWNER,source_contract=SOURCE,preparation=PREPARATION,model_source=REFERENCE_RUN+'/'+REFERENCE_ROW+'/source',source_RX=list(FIT),source_RX_holdout=None,L=6300,U_TX_hidden=True,U_used=False,V_statistically_used=False,target_access=False,model_seed=2026092701,checkpoint_rule='Existing source-final20/defaulthead;strict fullsource L6300/U56700/V27000 roles,no target selection',question='Does equal-expected-power band-constrained additive observation noise preserve fullsource identity and have source spectral support?',operator='Add stationary proper complex Gaussian noise only;C_ij=sinc(2*8.125MHz/25MHz*(i-j));PSD square root;never filter input IQ',nominal_SNR_dB=[20.,36.],noise_seed=20261017821,draws=[0,1],source_views=['raw','white_draw0','color_draw0','white_draw1','color_draw1'],source_NN=31500,synthetic_real_CP_smoke_NN=2,expected_total_NN=31502,new_model_epochs=0,optimizer_updates=0,teacher_forwards=0,head_fit_updates=0,Gaussian_amplitude_strictly_bounded=False,all_tail_samples_retained=True,criteria=dict(min_colored_accuracy_delta_pp=-.5,min_colored_macroF1_delta_pp=-.5,min_each_RX_TX_day_delta_pp=-1.,min_raw_correct_retention=.99,min_each_RX_median_source_inband_fraction=.90,min_median_relative_change=.01,max_relative_change=.15,require_each_fixed_draw=True),on_fail='Close fixed covariance/20..36dB/draws;no bandwidth/SNR/window/threshold/RX/seed/tail rescue',on_pass='Source identity and spectral-prior rationale only;separate preregistered matched fullsource E30 factorial required;not automatic training',new_Ag_efficacy_proved=False,As_efficacy_proved=False,joint_goal_complete=False,automatic_training=False,device='CPU',threads=2)
