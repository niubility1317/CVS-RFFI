"""Frozen local synthetic feasibility question,not an IQ training candidate."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
RUN='20261017-diagnostic-F-crossRX-jitterEstimator-synthetic-m1-r01'
OWNER='codex/root/F-sampling-jitter-estimator-20261017'
LOCAL='local_artifacts/F_sampling_jitter_estimator_20261017'
TEMPLATES=('multitone_quadratic','multitone_cubic','tone','constant','zero')
SNRS=(20.,28.,36.);DRAWS=32;PACKETS=70;CYCLES=4;PERIOD=20;DATA_SEED=202610174001
# Order: proper complex white noise,real gain,real phase,real sample jitter.
CASES=(('white',(1.,0.,0.,0.),0.),('gain_white',(.5,.5,0.,0.),0.),('phase_white',(.5,0.,.5,0.),0.),('gain_phase_white',(.34,.33,.33,0.),0.),('jitter_white',(.5,0.,0.,.5),0.),('all',(.4,.2,.2,.2),0.),('jitter_only',(0.,0.,0.,1.),0.),('CFOplus_white',(1.,0.,0.,0.),.0001),('CFOminus_white',(1.,0.,0.,0.),-.0001))

def config():
 return dict(run_id=RUN,owner=OWNER,output_root='WORKTREE/'+LOCAL+'/run',templates=list(TEMPLATES),SNR_dB=list(SNRS),cases=[dict(name=n,energy_shares=list(w),CFO_cycles_per_sample=f) for n,w,f in CASES],draws=DRAWS,packets_per_cell=PACKETS,cycles=CYCLES,period=PERIOD,data_seed=DATA_SEED,cells=4320,synthetic_packets=302400,synthetic_complex_points=24192000,estimator='All3two-pair partitions:independent-cycle cross outer-product bases;half squared pair difference covariance;unbiased cross Gram/right-hand moments;nonnegative4variance moment solve;no clean template input',derivative='20point periodic FFT derivative,all bins retained;not a deployed filter',arithmetic='Complex128/Float64',uncertainty_multiplier=5.,identifiability='Positive normalized crossGram min eigenvalue >5*Frobenius standarderror from70packet Grams;all diagonal entries positive',criteria=dict(min_identifiable_fraction_each_multitone_group=.95,min_rejected_fraction_each_degenerate_group=.95,zero_jitter_energy_fraction_median_max=.02,zero_jitter_energy_fraction_q95_max=.10,positive_jitter_relative_error_median_max=.20,positive_jitter_relative_error_q95_max=.50,nonCFO_covariance_relative_error_median_max=.20,nonCFO_covariance_relative_error_q95_max=.35),randomness='Same data_seed+template_index*10000+SNR_index*100+draw for all9cases;alllatent arrays drawn even when coefficient0;globalpacket phase random,percycle real gain/phase,perpoint real jitter/propernoise;templates fixed and normalized',scope='Synthetic-only first-order feasibility;no physicalsourceTX/RX/day or U roles;SNR/shares/CFO are stress tests,not future augmentation amplitude',cost=dict(real_IQ_reads=0,checkpoints=0,NN=0,updates=0,teacher_forwards=0,head_fit_updates=0,GPU=0,target_access=False),on_fail='Closefixedfour-cycle conditional-covariance jitter source entrance;no packet/cycle/window/bin/uncertainty/threshold/range/seed/oracle-template rescue;no realIQ/sourceexperiment/E20',on_pass='Sourcehypothesis remains unproved;separate source-only measurement preregistration needed;no automaticIQ access or training',baseline='F+CROSS_RX/3SIG_TIME/unifiedSinc/joint160/defaultcos/NoExplicitCFO/E20/originalU unchanged;no network in this diagnostic')
def validate(c):
 if c!=config():raise ValueError('Fixed synthetic-only configuration changed')
 return c

def write(p,value):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
