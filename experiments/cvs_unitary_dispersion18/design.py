"""One source-only identity-protection question;not a model-training candidate yet."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
RUN='20261018-diagnostic-F-unitaryDispersion-sourceV-s2026092701-m1-r01';RELEASE='cvs_F_unitary_dispersion_source_20261018_r01';LOCAL='local_artifacts/F_unitary_dispersion_source_20261018';OWNER='codex/root/F-unitary-dispersion-source'
BASE='PROJECT/runs/'+RUN
SOURCE='PROJECT/runs/phase1_daot_rc4_pure_game_m3_20260917_r2/source_contract.json'
PREPARATION='PROJECT/runs/20261011-phase1-stage2-geometry-source-manysig-m3-r01/s2026092701'
REFERENCE_RUN='20261017-phase1-F-crossRX-fullsource-concat-e20-s2026092701-m3-r02'
MODELS=('REPEAT','CONCAT');FIT=(1,3,4,6,8);PER_CELL=20;ORDINARY_SEED=20261018061
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
def resolve(alias):
 if not alias.startswith('PROJECT/'):raise ValueError('Registered project alias required')
 return Path(read(ROOT/'runtime.json')['project'])/alias[len('PROJECT/'):]
def config():
 return dict(run_id=RUN,launch_owner=OWNER,source_contract=SOURCE,preparation=PREPARATION,models=[dict(arm=a,row=a+'-s2026092701',source='PROJECT/runs/'+REFERENCE_RUN+'/'+a+'-s2026092701/source',epoch=20,selection='original fixed source-final20;not target-ranked',parent_updates=4440) for a in MODELS],source_RX=list(FIT),source_RX_holdout=None,L6300_unchanged=True,U_used=False,U_TX_hidden=True,V_role_count=27000,V_selected=1800,per_cell=PER_CELL,selection='first20 opaqueID-sorted V per6TXx5RXx3day;all90cells kept;no trainingdomainholdout',model_seed=2026092701,split_seed=392005,ordinary_seed=ORDINARY_SEED,base_views=['raw','ordinary'],ordinary='exact native parentargs/E13 schedule;one generated IQ reused across models/delta0/+-.1',operator=dict(kind='exp(i alpha L);proper complex linear finite-window dispersive observation',alpha_values=[0.,-.1,.1],max_abs_eigenphase_rad=.1,order=8,window=256,boundary='half-sample symmetric endpoints;nearest-neighbour generator;no circular wrap or outsideIQ',energy='unitary in exact arithmetic;Taylor8 remainder bound;source float error checked',extra_normalization=False,new_noise=False,TX_IQ_or_PA_added=False,source_RX_phase_estimated=False,physical_RX_replacement=False),criteria=dict(each_readout_base_endpoint_min_accuracy_delta_pp=-.5,each_RX_min_accuracy_delta_pp=-1.,each_TX_min_accuracy_delta_pp=-1.,each_readout_base_endpoint_correct_retention_min=.995,max_energy_relative_error=1e-5,max_inverse_relative_error=1e-5),expected_NN=21604,real_checkpoint_smoke_NN=4,real_checkpoint_loads=2,source_cache_loads=1,ordinary_generations=1800,operator_applications=21600,forward_operator_applications=10800,inverse_check_applications=10800,training_updates=0,teacher_NN=0,head_fit_updates=0,target_access=False,formal_test_changed=False,automatic_training=False,on_fail='Close this fixed profile/bound/order/boundary;no delta/profile/seed/parent rescue',on_pass='Necessary source protection only;no Ag/DG/As/joint efficacy or automatic training/promotion',old_negative_scope='Old phase estimator could not observe RX-associated phase in all source packets;not reopened,not used as evidence or fittedbasis;old amplitudeFIR/time-phase routes stayclosed',device='one shared-capacity GPU',threads=2)

def validate(c):
 if c!=config():raise ValueError('Fixed source roles/operator/budget/parents differ')
 return c
