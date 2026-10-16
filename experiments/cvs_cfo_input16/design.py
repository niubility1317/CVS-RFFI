"""One fixed no-training probe;not an identity result or permission for target."""
from pathlib import Path
from experiments.cvs_crossrx_satmix20 import design as prior
ROOT=Path(__file__).resolve().parents[2]
PROJECT,SOURCE,read,write=(getattr(prior,k) for k in ('PROJECT','SOURCE','read','write'))
RUN='20261016-diagnostic-F-crossRX-frequency-input-s2026092701-m1-r01'
RELEASE='cvs_F_crossRX_frequency_input_20261016_r01'
OWNER='codex/root/F-CROSS_RX-frequency-input-20261016'
LOCAL='local_artifacts/F_frequency_input16_20261016'
BASE=Path(PROJECT)/'runs'/RUN
BACKEND=PROJECT+'/releases/cvs_F_crossRX_satmix25_source_e20_20261016_r01'
BACKEND_COMMIT='0ae066adb10a001e834c8c679c5b7b81c20b3ac8'
SMOKE_SOURCE=Path(PROJECT)/'runs'/prior.RUN/'B00-RX1-s2026092701/source'
FIT=(3,4,6,8);AUDIT_SEED=20261016421;FREQUENCY_MAX=0.00010024013357398965
VIEWS=('identity','frequency_positive','frequency_negative','positive_inverse','negative_inverse')
def config():
 return dict(run_id=RUN,launch_owner=OWNER,method='Bounded native zero-origin received-frequency ramp,NoExplicitCFO retained',model='Fixed own F+CROSS_RX B00/seed2026092701/scratchE20/final20/4440 updates;not a parent for training',source_IQ_reads=0,target_access=False,training_updates=0,head_fit_updates=0,actual_NN_packets=162,synthetic_seed=AUDIT_SEED,synthetic_packets_per_view=32,views=list(VIEWS),frequency_max_cycles_per_sample=FREQUENCY_MAX,bound_basis='Actual ordinary aug_cfo_max .0004 times native E13 cap scale .2506003339349741',operator='x[n] exp(j2pi f n),n0..255;no centering,no added constant phase/RMS/noise/channel',NoExplicitCFO=True,internal_reference_compensation_retained=True,whole_network_frequency_invariance_assumed=False,on_fail='Close fixed frequency input hypothesis with0training,no frequency/seed/control rescue',on_pass='Engineering effect only;separately preregister source identity/E20 comparison before more IQ or training;no automatic target')
