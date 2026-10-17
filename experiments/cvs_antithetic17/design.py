"""One fixed full-source identity/observation guard;no target or training."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
RUN='20261017-diagnostic-F-antitheticNoise-sourceL-s2026092701-m1-r01'
RELEASE='cvs_F_antithetic_noise_source_20261017_r01'
LOCAL='local_artifacts/F_antithetic_noise_source_20261017'
OWNER='codex/root/F-antithetic-noise-source'
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
PROBE_SEED=20261018011
NOISE_SEED=20261018012
DROPOUT_SEED=20261018013
RECEIVER_SEED=20261016423
REPEATS=8
PER_CELL=2
def config():
 return dict(run_id=RUN,launch_owner=OWNER,source_contract=SOURCE,preparation=PREPARATION,model_source=REFERENCE_RUN+'/'+REFERENCE_ROW+'/source',source_RX=list(FIT),source_RX_holdout=None,source_parent_L=6300,diagnostic_L=180,selection='First2opaqueID-sorted visibleL packets in each6TXx5RXx3day cell;all5RX retained,no trainingholdout',batches='6fixedB30:one packet perTXxRX,perday andwithin-cellindex',U_used=False,U_TX_hidden=True,V_used=False,target_access=False,model_seed=2026092701,split_seed=392005,probe_seed=PROBE_SEED,noise_seed=NOISE_SEED,dropout_seed=DROPOUT_SEED,receiver_seed=RECEIVER_SEED,scenes=list(SCENES),repeats=REPEATS,per_cell=PER_CELL,question='Same fullpractical Gaussian noise marginals:does e/-e coupling reduce existing RDrop gradient variance vs e/e2 at matching2views?',route='original complete residual/post_sync/NoEQ/referenceSNR30/reflect/256/25MHz',coupling='Same H,geometry,state,hardware,quality,compensation;replace only standardNormal noise driver before full receiver/AGC. Independently recompute eachAGC;never averageIQ',RDrop='Two forwards same generatedIQ perview;alpha1/LS.01;CRN slot2 masks shared across independent/antithetic comparison;slot1/slot2 masks independent',gradient_object='Existing full trainable parameters in native storedcoordinates,plus unchanged defaultcosine W6x160;no optimizer,headfit,teacher or newloss',raw_units_limitation='Fullgradient trace is parameter-coordinate dependent;defaulthead reported separately;not AdamW update variance or identity gain',expected_NN=25922,expected_backward_calls=432,expected_channel_calls=12960,real_checkpoint_smoke_NN=2,real_checkpoint_loads=1,source_cache_loads=1,training_updates=0,teacher_NN=0,head_fit_updates=0,formal_test_changed=False,automatic_training=False,criteria=dict(overall_full_gradient_variance_ratio_max=.8,overall_default_head_variance_ratio_max=.8,each_mid_low_full_gradient_variance_ratio_max=1.),on_fail='Close fixed coupling for this recipe;no draw/scene/seed/threshold rescue',on_pass='Only source estimator evidence;matching fullsource E30 training required separately,no identity or joint success claim',physical_basis='Centered Gaussian is symmetric;eachview preserves original noise law,not sameview denoising',reference='https://proceedings.mlr.press/v89/wu19c.html',reference_limit='Variational-inference evidence motivates variance hypothesis only;not a CVS training result',device='one shared-capacityGPU',threads=2)
