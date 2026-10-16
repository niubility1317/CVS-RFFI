"""One frozen source phase-curvature association question;0 models or IQ augmentation."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
RUN='20261017-diagnostic-F-crossRX-frequencyPhase-fitL-manysig-m1-r01';RELEASE='cvs_F_frequency_phase_fitL_20261017_r01';OWNER='codex/root/F-frequency-phase-source-basis-20261017';LOCAL='local_artifacts/F_frequency_phase_source_basis_20261017'
SOURCE='PROJECT/runs/phase1_daot_rc4_pure_game_m3_20260917_r2/source_contract.json';PREPARATION='PROJECT/runs/20261011-phase1-stage2-geometry-source-manysig-m3-r01/s2026092701'
BASE='PROJECT/runs/'+RUN;FIT=(3,4,6,8);WINDOW=(80,160);DIM=8

def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
def resolve(alias,runtime):
 if not alias.startswith('PROJECT/'):raise ValueError('Registered projectrelative handle required')
 return Path(runtime['project'])/alias[len('PROJECT/'):]

def config():
 return dict(run_id=RUN,launch_owner=OWNER,output_root=BASE,source_contract=SOURCE,preparation=PREPARATION,fit_L=5040,fit_RX=list(FIT),days=[1,2,3],TX=list(range(6)),samples_per_cell=70,IQ_length=256,window=list(WINDOW),sample_rate_Hz=25000000,arithmetic='Originalfloat IQ toComplex128;no newnormalization orIQview',measurement_dependency='experiments.cvs_frequency_phase17.estimator.observe/summarize;unchanged synthetic-qualified code',nominal_period=20,harmonics=[-6,-5,-4,-3,-2,-1,1,2,3,4,5,6],closure_centers=[-5,-4,-3,-2,2,3,4,5],CFO='Blindlag20 reference correction only;estimate discarded from phaseidentitycoordinates;no oracle ornewmodelcompensation',carrier_noise_ratio_min=25.,cell_resultant_min=.8,RX_reference_resultant_min=.8,relative_numerical_floor=1e-12,all_5040_packets_required=True,all_72_cells_required=True,observation='8 unitadjacentharmonicphaseclosures;all8centers andall70packets perTX/day/RX cell;circular RXreference andRXarithmeticcentering',crossfit='18folds exclude BOTH heldTX andheldday;5TX*2daycellmeans fit4RX;zeroprofileand3cyclicwrongRX controls;no classifier orfittedscale',criteria=dict(profile_RMS_min_rad=.02,crossfit_risk_improvement_min=.5,better_folds_min=15,each_RX_not_worse=True,correct_minus_best_wrong_RX_min=.05,all_packets_identifiable=True,all_cell_RX_resultants_identifiable=True),on_fail='Closefixedsourcephasebasis;no measure/window/period/harmonic/CFO/quality/threshold/TX/day/RX/packet/seed rescue;no IQaugmentation orE20',on_pass='SourceRX-associated phase evidence only;separate physicallyjustified boundedIQ operator/identityprotection andmatchedE20 still required;no automaticaugmentation/training',source_only=True,model_constructed=False,checkpoint_loads=0,source_cache_loads=1,real_IQ_statistically_used=5040,observer_calls=1,network_forwards=0,optimizer_steps=0,head_fit_updates=0,teacher_forwards=0,IQ_views_generated=0,IQ_modified=False,U_truth_read=False,U_statistically_used=False,V_statistically_used=False,held_RX1_statistically_used=False,target_access=False,automatic_training=False,device='CPU',threads=2,NoExplicitCFO_preserved=True,phase_or_CFO_not_identity_feature=True,baseline_boundary='F+CROSS_RX/3SIG_TIME/unifiedSinc/joint160/defaultcos/E20/originalU unchanged;no model used here',limitations=['Finitewindowharmonicphase includesTX/channel/RX/equalizer effects;not pureRXhardware causation','Syntheticqualified observerdoesnot guarantee realEQ1periodicity orcarriercontent;no invalidpackets dropped','Sourceassociation isnotIQlabelpreservation,receiverreplacement,heldRXrecognition,DG,As orjointproof','ExistingcacheL6300/unusedV/features deserialize;onlyfitL5040 statisticallyused;UopaqueIDs only','Not oldamplitudespectralFIR/timepointphase/cyclecov/jittervariance rescue;none reopened'])
def validate(c):
 if c!=config():raise ValueError('Fixed source phase roles/observation/threshold/budget differ')
 return c
