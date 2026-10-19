import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
RUN='20261019-diagnostic-F-cleanLookahead-sourceL-s2026092701-m2-r01'
RELEASE='cvs_F_cleanLookahead_source_20261019_r01'
LOCAL='local_artifacts/F_cleanLookahead_source_20261019';OWNER='codex/root/F-clean-lookahead'
REFERENCE_RELEASE='cvs_F_nontrue_CSIL_E20_20261019_r01';REFERENCE_COMMIT='1e7cc414f73f093dfbdccee0e4ba7cf90c7a5324'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
PROJECT=read(ROOT/'runtime.json')['project'] if (ROOT/'runtime.json').is_file() else 'PROJECT'
BASE=Path(PROJECT)/'runs'/RUN
SCENES=('practical_mid_urban','practical_low_urban');VIEWS=('raw',*SCENES)
SEED=2026092701;CHANNEL_SEED=20261019624;RECEIVER_SEED=20261019625
def config():return dict(schema='F_clean_lookahead_source_v1',run_id=RUN,launch_owner=OWNER,model_seed=SEED,models=['B0','B1'],reference_run='20261019-phase1-F-nontrue-CSIL-e20-s2026092701-m4-r01',checkpoint='strict own scratch FINAL20/defaulthead; original student unchanged',source_role='L_s only',source_counts=dict(L=6300,U=56700,V=27000),source_RX=[1,3,4,6,8],source_days=[1,2,3],data_split_seed=392005,selection='First2 original cache entries per TX/RX/day for step; next2 for response; both historically seen L, never independent held-out validation',step_unique=180,response_unique=180,batch=30,batches_per_model=6,views=list(VIEWS),observation_namespace='F_cleanLookahead/sourceL/draw0',channel_seed=CHANNEL_SEED,receiver_seed=RECEIVER_SEED,channel='raw to complete residual/post_sync/noEQ/ref30/reflect; native quality/sync/IQ compensation/AGC recomputed; no ordinary stacked',ordinary_augmentation=False,model_mode='eval with autograd; original .grad None; not stochastic training reconstruction',virtual_optimizer=dict(type='fresh AdamW',lr=.0002,weight_decay=.0001,betas=[.9,.999],eps=1e-8,clip_physical=5.,Sinc='CENTER_UNITS'),counterfactuals=['RAW_STEP','JOINT_STEP','FIRST_ORDER_LOOKAHEAD'],joint_gradient='.5 g_raw+.5 mean(g_mid,g_low)',lookahead_gradient='.5 g_raw(theta)+.5 mean(g_mid,g_low)(theta_after_raw_step); detached first order; no Hessian term',default_head='existing jointly updated cosine6x160 scale30; no new head',expected_NN=6188,expected_virtual_steps=42,persistent_original_updates=0,new_trained_models=0,synthetic_real_checkpoint_smoke_NN=68,expected_network_backward_packets=1820,target_access=False,U_truth_read=False,V_values_used=False,qualification=dict(rule='For each B0/B1, lookahead beats joint mean response mid/low CE by >1e-6; response raw CE no worse than joint by >1e-6; at least4/6 response batches improve urban CE; only mechanistic support, not performance/E50',automatic_training=False,automatic_E50=False,on_fail='close this fixed fresh-Adam raw-lookahead rationale; no step-size/inner-state/seed rescue'))
def validate(c):
 if c!=config():raise ValueError('Unregistered source/view/step/provenance matrix')
 return c
