"""One source-only identity-protection question;not a model-training candidate yet."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
RUN='20261018-diagnostic-F-repeatShrink-sourceV-s2026092701-m1-r01';RELEASE='cvs_F_repeat_shrink_source_20261018_r01';LOCAL='local_artifacts/F_repeat_shrink_source_20261018';OWNER='codex/root/F-repeat-shrink-source'
BASE='PROJECT/runs/'+RUN
SOURCE='PROJECT/runs/phase1_daot_rc4_pure_game_m3_20260917_r2/source_contract.json'
PREPARATION='PROJECT/runs/20261011-phase1-stage2-geometry-source-manysig-m3-r01/s2026092701'
REFERENCE_RUN='20261017-phase1-F-crossRX-fullsource-concat-e20-s2026092701-m3-r02'
MODELS=('REPEAT','CONCAT');FIT=(1,3,4,6,8);PER_CELL=20;ORDINARY_SEED=20261018101
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
def resolve(alias):
 if not alias.startswith('PROJECT/'):raise ValueError('Registered project alias required')
 return Path(read(ROOT/'runtime.json')['project'])/alias[len('PROJECT/'):]
def config():
 return dict(run_id=RUN,launch_owner=OWNER,source_contract=SOURCE,preparation=PREPARATION,models=[dict(arm=a,row=a+'-s2026092701',source='PROJECT/runs/'+REFERENCE_RUN+'/'+a+'-s2026092701/source',epoch=20,selection='original fixed source-final20;not target-ranked',parent_updates=4440) for a in MODELS],source_RX=list(FIT),source_RX_holdout=None,L6300_unchanged=True,U_used=False,U_TX_hidden=True,V_role_count=27000,V_selected=1800,per_cell=PER_CELL,selection='first20 opaqueID-sorted V per6TXx5RXx3day;all90cells kept;no trainingdomainholdout',model_seed=2026092701,split_seed=392005,ordinary_seed=ORDINARY_SEED,base_views=['raw','ordinary'],ordinary='exact native parentargs/E13 schedule;one generated IQ reused across models/strength0and.1',operator=dict(kind='own-packet four-cycle residual contraction;proper complex linear;not denoising ground truth',strength_values=[0.,.1],max_strength=.1,window=[80,160],period=20,cycles=4,taper='same sin(pi*t/19)^2 for t0..19 in each cycle;endpoints exactly0',equation='x_W - strength*w*(x_W - mean_cycle(x_W))',boundary='outside80:160 and each20point cycle first/last sample bitwise unchanged;no wrap/padding/outsideIQ',normalization='no extra RMS normalization;fixed cycle mean retained,nonperiodic component contracted',new_noise=False,CFO_estimated=False,TX_IQ_or_PA_added=False,labels_used_by_operator=False,physical_RX_replacement=False),criteria=dict(each_readout_base_endpoint_min_accuracy_delta_pp=-.5,each_RX_min_accuracy_delta_pp=-1.,each_TX_min_accuracy_delta_pp=-1.,each_readout_base_endpoint_correct_retention_min=.995,max_energy_increase_relative=1e-6,max_cycle_mean_change_relative=1e-6,max_total_relative_change=.100001,outside_unchanged=True,cycle_boundaries_unchanged=True),expected_NN=14404,real_checkpoint_smoke_NN=4,real_checkpoint_loads=2,source_cache_loads=1,ordinary_generations=1800,operator_applications=7200,nonzero_operator_applications=3600,training_updates=0,teacher_NN=0,head_fit_updates=0,target_access=False,formal_test_changed=False,automatic_training=False,on_fail='Close fixed period20/window80:160/taper/strength.1;no range/window/taper/seed/parent rescue',on_pass='Necessary source protection only;no Ag/DG/As/joint efficacy;separate matched E30 training still required',old_negative_scope='Residual-covariance RX attribution,periodic-projector RX attribution,cycle permutation,unitary phase,external donor coherent mix remain closed;this operator fits no RX profile and introduces no donor',device='one shared-capacity GPU',threads=2)

def validate(c):
 if c!=config():raise ValueError('Fixed source roles/operator/budget/parents differ')
 return c
