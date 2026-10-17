"""One fixed all-source temporal power-envelope basis;no classifier or training."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
RUN='20261017-diagnostic-F-envelopeProfile-fullL-manysig-m1-r01'
RELEASE='cvs_F_envelope_profile_source_20261017_r01'
LOCAL='local_artifacts/F_envelope_profile_20261017'
OWNER='codex/root/F-envelope-profile-source'
BASE='PROJECT/runs/'+RUN
SOURCE='PROJECT/runs/phase1_daot_rc4_pure_game_m3_20260917_r2/source_contract.json'
PREPARATION='PROJECT/runs/20261011-phase1-stage2-geometry-source-manysig-m3-r01/s2026092701'
FIT=(1,3,4,6,8)
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
def resolve(alias):
 if not alias.startswith('PROJECT/'):raise ValueError('Registered project alias required')
 return Path(read(ROOT/'runtime.json')['project'])/alias[len('PROJECT/'):]
def config():
 return dict(run_id=RUN,launch_owner=OWNER,source_contract=SOURCE,preparation=PREPARATION,source_RX=list(FIT),source_RX_holdout=None,L=6300,U_used=False,U_TX_truth_hidden=True,V_used=False,target_access=False,question='Does post-RMS slow temporal power shape contain an RX-associated component transferable across TX and day?',observation=dict(samples=256,blocks=16,samples_per_block=16,relative_floor=1e-6,per_packet_power_normalized=True,center_log_profile=True,DCT_modes=[1,2],observation_window='all256,no crop/shift/window selection'),crossfit='18 folds exclude one TX AND one day from statistics fit;all5RX retained in every fold;this is not a model training split',control='Fixed cyclic RX permutation by1 of predicted effects;no control choice',criteria=dict(relative_prediction_MSE_max=.9,better_folds_min=15,structured_better_than_wrong_RX=True,each_RX_relative_MSE_lt=1.),on_fail='Close fixed temporal-envelope basis;no block/mode/floor/window/RX/seed/threshold rescue',on_pass='RX association only,not AGC causality or Ag efficacy;separate bounded IQ and default-head identity study needed before any matched E30 training',new_IQ_views=0,checkpoint_loads=0,NN=0,teacher_forwards=0,training_updates=0,head_fit_updates=0,source_cache_loads=1,source_IQ_unique=6300,CPU_threads=2,GPU='none',automatic_training=False,training_budget_if_later_justified='Latest E30,F+crossRXsameTX/all5RX/defaulthead/originalU;not authorized by this source statistic alone')
