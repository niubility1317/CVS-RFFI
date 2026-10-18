"""Two fixed P43 FINAL30 students; descriptive source diagnosis only."""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
RUN='20261019-diagnostic-FcrossRX-current-controls-e30-s2026092701-m2-r01'
RELEASE='cvs_FcrossRX_current_controls_E30_20261019_r01'
OWNER='codex/root/current-controls79'
REFERENCE_RUN='20261018-phase1-FcrossRX-impulse-factorial-e30-s2026092701-m6-r01'
REFERENCE_RELEASE='cvs_FcrossRX_impulse_factorial_E30_20261018_r01'
REFERENCE_COMMIT='19aedc3604fed9c193237dd92366164178af0983'
PANEL_RUN='20261017-diagnostic-FcrossRX-fullsource-residual-s2026092701-m3-r01'
HELPER_PACKAGE='cvs_fullsource_residual_diag24'
SEED=2026092701
ARMS=('B00','B01')
LABELS=('NOAS30','AS30')
RX=(1,3,4,6,8)
DAYS=(1,2,3)
SCENES=('practical_low_suburban','practical_mid_urban','practical_low_urban')
CONDITIONS=('full','no_noise/fixed','no_noise/total','flat_H/fixed','flat_H/total','no_carrier/fixed','no_carrier/total','no_RX_IQ/fixed','no_RX_IQ/total','actual_power_quality','edge_boundary')
PHYSICS_FIELDS=('quality_snr_db','white_proxy_quality_db','actual_waveform_quality_db','waveform_minus_white_power_db','output_frequency_hz','estimated_frequency_hz','signal_to_added_noise_db','noise_coloration_dynamic_range_db','agc_gain_db','agc_limited','residual_kernel_energy','kernel_length','numerical_delay','interior_count','residual_scale','lock','state')
N=1800
DRAWS=(0,1)
BATCH=GRAD_N=90
IQ_SHAPE=(2,3,11,N,2,256)
PHYSICS_SHAPE=IQ_SHAPE[:4]
BRANCHES=('time','frequency','PA','reference','signature')
BRANCH_METRICS=('clean_sat_cosine','clean_norm','sat_norm')
GRADIENT_GROUPS=('Sinc_Hz','signature','reference','default_head','PA','time','frequency','remaining','whole')
PREDICTION_NN=N*(1+2*3*11)
GRADIENT_NN=GRAD_N*(1+2*3)
SMOKE_NN=2
ROW_NN=PREDICTION_NN+GRADIENT_NN+SMOKE_NN
TOTAL_NN=2*ROW_NN
FP32=dict(cudnn_allow_tf32=False,cuda_matmul_allow_tf32=False,cudnn_benchmark=False,cudnn_deterministic=False,matmul_precision='highest')

def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def write(path,value):
 Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')

def paths(project):
 project=Path(project)
 return dict(project=project,output=project/'runs'/RUN,reference_release=project/'releases'/REFERENCE_RELEASE,panel=project/'runs'/PANEL_RUN/'diagnostic')

def source(project,arm):
 if arm not in ARMS:raise ValueError('Only fixed original P43 B00/B01 students')
 return Path(project)/'runs'/REFERENCE_RUN/(arm+'-s'+str(SEED))/'source'

def config(project,device='cuda:0'):
 p=paths(project)
 return dict(run_id=RUN,owner=OWNER,reference_run=REFERENCE_RUN,reference_commit=REFERENCE_COMMIT,model_seed=SEED,model_seed_count=1,checkpoint_state='model',EMA_loaded=False,rows=[dict(label=label,arm=arm,row_id=arm+'-s'+str(SEED),checkpoint=str(source(project,arm)/'final_ssdg.pth')) for arm,label in zip(ARMS,LABELS)],panel_root=str(p['panel']),panel_role='V',source_counts=dict(L_s=6300,U_s=56700,V=27000),source_RX=list(RX),source_days=list(DAYS),panel_count=N,draws=list(DRAWS),scenes=list(SCENES),conditions=list(CONDITIONS),IQ_shape=list(IQ_SHAPE),device=device,batch_size=BATCH,gradient_samples=GRAD_N,prediction_NN_per_row=PREDICTION_NN,gradient_NN_per_row=GRADIENT_NN,smoke_NN_per_row=SMOKE_NN,expected_NN_per_row=ROW_NN,expected_NN=TOTAL_NN,gradient_scope='90 clean plus 90 per draw/scene full-condition eval-mode smoothedCE; not stochastic training gradients',head='existing joint default cosine6x160 scale30',training_epochs=0,optimizer_updates=0,EMA_updates=0,head_fit_updates=0,new_IQ_generated=0,new_channel_draws=0,U_iteration=False,U_truth_read=False,query_access=False,formal_scores_read=False,source_promotion=False,automatic_training=False,automatic_formal=False,automatic_budget_extension=False,formal_test_changed=False,scientific_scope='descriptive matched frozen-source controls; no candidate selection from control gains')
