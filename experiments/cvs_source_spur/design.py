"""Fixed non-DC narrow-line source hypothesis; no model or IQ augmentation."""
from pathlib import Path
from experiments.cvs_source_repeatcorr import design as previous
ROOT=Path(__file__).resolve().parents[2]
PROJECT,SOURCE,PREPARATION,REFERENCE,FIT,TX,DAYS,read,write=(getattr(previous,k) for k in ('PROJECT','SOURCE','PREPARATION','REFERENCE','FIT','TX','DAYS','read','write'))
RUN='20261017-diagnostic-source-nonDC-spur-crossfit-manysig-m1-r01'
RELEASE='cvs_source_nonDC_spur_crossfit_20261017_r01'
OWNER='codex/root/source-nonDC-spur-20261017'
BASE=Path(PROJECT)/'runs'/RUN
FS=25000000.;BINS=tuple(i for i in range(8,248) if abs(i-128)>8)
def rows():return [dict(row_id='nonDC-line-fitRX3_4_6_8',arm='nonDC_line')]
def config(row):
 if row not in rows():raise ValueError('Only one registered source line hypothesis')
 return dict(schema='source_nonDC_line_crossfit_v1',run_id=RUN,launch_owner=OWNER,**row,output_root=(BASE/row['row_id']).as_posix(),source_contract=SOURCE,source_preparation=PREPARATION,reference_metadata=REFERENCE,source_RX=list(FIT),source_TX=list(TX),source_days=list(DAYS),L_packets=5040,packets_per_cell=70,held_RX1_used=False,U_used=False,V_used=False,target_access=False,source_only=True,model_forwards=0,checkpoint_loads=0,optimizer_updates=0,new_models=0,head_fit_updates=0,channel_generation=False,device='CPU',CPU_threads=2,CUDA_VISIBLE_DEVICES='',automatic_training=False,
 observation=dict(IQ_length=256,fs_hz=FS,FFT_bins=256,zero_padding=False,windows=['rectangular','hann'],packet_power_normalization='Periodogram divided by its mean over all256 bins; analysis only,not changedIQ',cell_PSD='Mean of all70 normalized periodograms in each6TX x3day x4RX cell',candidate_shifted_bins=list(BINS),DC_excluded_bins='120..136',edge_exclusion='0..7 and248..255',line_power='Mean of bins k-1,k,k+1',background='Median of8 bins k+-4..7',prominence='10log10(line/background),floor relative1e-12',resolution_claim='Three-bin local excess at97.65625kHz/bin; not physical narrow-line resolution or pureRX hardware'),
 crossfit='18 folds exclude oneTX AND one day from fit; train other5TX x2days; selection on rectangular fit only,all held values excluded',
 selector='Fixed maximum positive rectangular RX contrast over4RX and223candidate bins; same selectedRX/bin evaluated in BOTH windows; no test selection',
 controls='Zero RX contrast and3cyclic RX permutations at the same fit-selected bin',
 criteria=dict(train_prominence_min_db=6.,train_RX_contrast_min_db=3.,held_prominence_min_db=3.,held_RX_contrast_min_db=3.,successful_folds_min=15,modal_exact_RX_bin_folds_min=15,each_window_risk_reduction_min=.5,each_window_positive_folds_min=15,each_window_wrong_RX_gap_min=.2,each_TX_day_RX_risk_reduction_positive=True),
 on_fail='Close fixed nonDC local-line hypothesis;no FFT/window/background/DC/bin/threshold/RX/TX/day rescue',on_pass='Source association only;separate bounded same-power IQ intervention required before any scratchE20,not pureRX orAg orlabel-safety proof',fixed_baseline='F+CROSS_RX sameTX/SIG_TIME_X3/unifiedSinc/NoExplicitCFO/joint160/defaultcos30/E20;no model here',formal_test_changed=False,goal_complete=False)
def validate(c):
 if c!=config(rows()[0]):raise ValueError('Fixed line/source/no-model config differs')
