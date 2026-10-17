"""One fixed full-source identity/observation guard;no target or training."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
RUN='20261017-diagnostic-F-N0prior-fullL-s2026092701-m1-r01'
RELEASE='cvs_F_N0_prior_source_20261017_r01'
LOCAL='local_artifacts/F_N0_prior_source_20261017'
OWNER='codex/root/F-N0-prior-source'
BASE='PROJECT/runs/'+RUN
SOURCE='PROJECT/runs/phase1_daot_rc4_pure_game_m3_20260917_r2/source_contract.json'
PREPARATION='PROJECT/runs/20261011-phase1-stage2-geometry-source-manysig-m3-r01/s2026092701'
REFERENCE_RUN='20261017-phase1-F-crossRX-fullsource-concat-e20-s2026092701-m3-r02'
REFERENCE_ROW='CONCAT-s2026092701'
FIT=(1,3,4,6,8)
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
def resolve(alias):
 if not alias.startswith('PROJECT/'):raise ValueError('Registered project alias required')
 return Path(read(ROOT/'runtime.json')['project'])/alias[len('PROJECT/'):]
SCENES=('practical_low_suburban','practical_mid_urban','practical_low_urban')
PROBE_SEED=20261017931
PROBE_NAMESPACE='MMSE_source_coupled'  # Keep existing source latent draws;MMSE is always disabled here.
RECEIVER_SEED=20261016423
VARIANTS=('nominal','N0_half','N0_1p5','N0_half_fixedQ','N0_1p5_fixedQ')
FACTORS={'nominal':1.,'N0_half':.5,'N0_1p5':1.5,'N0_half_fixedQ':.5,'N0_1p5_fixedQ':1.5}
VIEWS=('raw',)+tuple(s+'/'+k for s in SCENES for k in VARIANTS)
def config():
 return dict(run_id=RUN,launch_owner=OWNER,source_contract=SOURCE,preparation=PREPARATION,model_source=REFERENCE_RUN+'/'+REFERENCE_ROW+'/source',source_RX=list(FIT),source_RX_holdout=None,L=6300,U_TX_hidden=True,U_used=False,V_statistically_used=False,target_access=False,model_seed=2026092701,checkpoint_rule='Fixed ordinary fullAs source-final20;all6300/56700/27000;no target-based choice',question='Does a fixed equalprior .5N0/1.5N0 complete-practical source observation law provide useful identity observations at the same mean reference noise power?',probe_seed=PROBE_SEED,probe_namespace=PROBE_NAMESPACE,receiver_seed=RECEIVER_SEED,scenes=list(SCENES),views=list(VIEWS),sample_rate_Hz=25000000.,carrier_Hz=2462000000.,reference_snr_db=30.,route='residual/post_sync',equalization_enabled=False,noise_prior=dict(factors=[.5,1.5],probabilities=[.5,.5],mean_linear_N0_factor=1.,interpretation='Engineering sensitivity law,not measured temperature distribution;mean SNR in dB is not preserved',location='Replace original reference N0 before receiver_quality;no second AWGN injection',recomputed='Original lock/fineCFO/phase-tracking/residual scale and AGC recomputed with changedN0',fixedQ_controls='Only causal controls hold nominal quality;never training candidates'),shared='Same H/geometry/state/shadow/scatter/receiver/phase/tracking and standard normal noise draws;only N0 and consequent receiver processing differ',boundary='Originalreflect;same IQ256/no crop/shift/extraRMS',expected_NN=100802,source_decisions=100800,physical_realizations=18900,processing_calls=94500,new_model_epochs=0,training_updates=0,teacher_forwards=0,head_fit_updates=0,criteria=dict(min_expected_low_gain_pp=.5,min_expected_mid_gain_pp=.5,min_expected_suburban_gain_pp=-.25,min_each_RX_expected_low_gain_pp=-1.),on_fail='Close fixed .5/1.5 N0 prior;no SNR/factor/probability/scene/seed/threshold rescue',on_pass='Source input evidence only;no Ag/As training gain or joint proof;matched fullsource E30 training separately required',formal_test_changed=False,automatic_training=False,device='CPU',threads=2,physical_reference='https://www.mathworks.com/help/comm/ref/receiverthermalnoise.html',limitation='Thermal-noise model motivates variable power only;no published CVS benefit or empirical choice of these factors is claimed')
