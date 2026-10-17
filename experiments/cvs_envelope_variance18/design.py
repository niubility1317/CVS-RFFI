"""One reanalysis of existing legal source observations;no new IQ or model."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
RUN='20261018-diagnostic-F-envelopeVariance-sourceL6300-m1-r01';RELEASE='cvs_F_envelope_variance_source_20261018_r01';LOCAL='local_artifacts/F_envelopeVariance_sourceL6300_20261018';OWNER='codex/root/F-envelope-variance-source'
BASE='PROJECT/runs/'+RUN
SOURCE='PROJECT/runs/phase1_daot_rc4_pure_game_m3_20260917_r2/source_contract.json'
OBSERVATIONS='PROJECT/runs/20261017-diagnostic-F-envelopeProfile-fullL-manysig-m1-r01'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
def runtime():return read(ROOT/'runtime.json' if (ROOT/'runtime.json').is_file() else ROOT/LOCAL/'runtime.json')
def resolve(p):
 if not p.startswith('PROJECT/'):raise ValueError('Registered project alias required')
 return Path(runtime()['project'])/p[len('PROJECT/'):]
def config():
 return dict(run_id=RUN,source_contract=SOURCE,observations=OBSERVATIONS,launch_owner=OWNER,question='Does within-TX/RX/day envelope variability have a shared RX scale pattern transferable across TX and day without a strong TX-specific advantage?',original_observation='Reuse original full256/16x16/logpowercentered/DCT1;DCT2 not a fallback;no new IQ measurement',scale='Peroriginal70packetcell median absolute deviation from cellmedian,no sample deletion;logMAD floor1e-12',crossfit='18fold exclude TX AND day for shared RX pattern;all5RX always;TXspecific control fits sameTX/other2days;not a trainingdomainholdout',criteria=dict(RX_relative_MSE_max=.9,better_folds_min=15,each_RX_relative_MSE_max=1.,RX_better_than_fixed_wrong_RX=True,RX_SSE_at_most_1_05_TX_specific_SSE=True),on_fail='Close fixed DCT1/within-cellMAD/RXvariability source hypothesis;no mode/quantile/group/RX/threshold rescue',on_pass='Source scale evidence only;not AGC causality,identitypreservation or DG efficacy. Separate bounded IQ/defaulthead evidence and registered E30 controls required',L=6300,source_RX=[1,3,4,6,8],source_RX_holdout=None,U_used=False,U_TX_truth_hidden=True,V_used=False,target_access=False,new_IQ_reads=0,checkpoint_loads=0,NN=0,updates=0,teacher_forwards=0,new_IQ_views=0,automatic_training=False,CPU_threads=2,GPU='none',F_training_budget_unchanged='E30/F+CROSS_RX/defaulthead/originalU if later separately justified')
