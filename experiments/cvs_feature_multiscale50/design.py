from pathlib import Path
from experiments.cvs_legacy_no_sat import design as legacy
ROOT=Path(__file__).resolve().parents[2]
PROJECT,SOURCE,CAPSULE,TRUTH,CLASSES,SEEDS,FULL_FP32_POLICY,read,write=(getattr(legacy,k) for k in ('PROJECT','SOURCE','CAPSULE','TRUTH','CLASSES','SEEDS','FULL_FP32_POLICY','read','write'))
RUN='20261013-phase1-feature-multiscale-oldgate-cos-e50-clean-manysig-m12-r01'
RELEASE='cvs_feature_multiscale50_20261013_r01'
OWNER='codex/root/feature-multiscale50-oldgate-20261013'
BASE=Path(PROJECT)/'runs'/RUN
ARMS=('OLD_GATE_COS50','TIME_MS50','PA_MS50','JOINT_MS50')
VIEWS=('clean',);READOUTS=('NATIVE_HEAD',);EPOCHS=50;LABEL_EPOCHS=33;PSEUDO_EPOCHS=17;STEPS=222;UPDATES=11100
TIME_LAYERS=('t1','t2','t3');PA_LAYERS=('pa_b1','pa_b2','pa_b3')
def rows():return [dict(row_id=f'{a}-s{s}',arm=a,model_seed=s) for s in SEEDS for a in ARMS]
def legacy_reference(seed):return legacy.config(next(r for r in legacy.rows() if r['model_seed']==seed))
def config(row):
 if row not in rows():raise ValueError('Unregistered feature extractor row')
 a=row['arm'];seed=row['model_seed'];ref=legacy_reference(seed)
 return dict(ref,schema='feature_multiscale_oldgate50_v1',run_id=RUN,**row,output_root=(BASE/row['row_id']/'source').as_posix(),launch_owner=OWNER,epochs=EPOCHS,label_epochs=LABEL_EPOCHS,pseudo_epochs=PSEUDO_EPOCHS,updates=UPDATES,legacy_reference_config=ref ,source_falsification=dict(first_seed=SEEDS[0],maximum_final_L_CE=.5,metric='mean train/loss_tx_labeled over final222updates; no V or target feedback'),training_recipe='historical batch_neighbor pseudo gate+EMA+cos; no satellite; allarms identical',baseline_definition='user explicitly confirmed historical pseudo gate+EMA+cos, not CE-only baseline',explicit_CFO_identity_feature=False,feature_extractor=dict(time=a in ('TIME_MS50','JOINT_MS50'),PA=a in ('PA_MS50','JOINT_MS50'),time_layers=list(TIME_LAYERS),PA_layers=list(PA_LAYERS),time_kernel_dilation=[[3,1],[5,1],[5,2]],PA_kernels=[3,5,7],PA_dilation_by_layer=[1,2,4],residual='learned1x1 shortcut before original norm/activation/pool/dropout',output_embedding=160),classifier='unchanged baseline ResidualPhysicalHead/6x160 cosine/scale30/branchLN/gain; no newhead/readout/calibration',test_views=list(VIEWS),checkpoint_selection='final_only',goal=dict(primary='JOINT_MS50',clean_three_seed_mean=.84,feedback_to_training=False),budget_authorization='user: feature extraction, historicaloldgate+EMA+cos; Agentchooseepochs normally<=50/model; fixedscratch50 33label+17pseudo allcontrols',cost_claim='same50epochs/11100 updates andnative L128/U256 recipe; extractor parameters/actualforward time measured separately')
def validate(c):
 row=next((r for r in rows() if r['row_id']==c.get('row_id')),None)
 if row is None or c!=config(row) or c['epochs']>50:raise ValueError('Fixed oldgate feature config differs')
 return c
def make_args(c,device='cuda:0'):
 validate(c);a=legacy.make_args(c['legacy_reference_config'],device)
 a.output_dir=c['output_root'];a.run_id=c['run_id'];a.candidate_id=c['row_id'];a.seed=c['model_seed'];a.epochs=EPOCHS;a.label_epochs=LABEL_EPOCHS;a.pseudo_epochs=PSEUDO_EPOCHS
 a.checkpoint_selection='final_only';a.test_eval_start_epoch=999999
 return a
