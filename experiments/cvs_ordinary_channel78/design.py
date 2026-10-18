"""Only bypass L ordinary AWGN/MP in the original P43 final ten epochs."""
from copy import deepcopy
from pathlib import Path
from experiments.cvs_lraw_tail30 import design as old
ROOT=Path(__file__).resolve().parents[2]
RUN='20261019-phase1-FcrossRX-ordinaryChannel-e30-s2026092701-m2-r01'
RELEASE='cvs_FcrossRX_ordinaryChannel_E30_20261019_r01'
OWNER='codex/root/ordinary-channel78';LOCAL='local_artifacts/cvs_FcrossRX_ordinaryChannel_E30_20261019_r01'
DEPENDENCY_COMMIT='d14450bf1b0d12cc281df1f4d0e79e8760e48282'
REFERENCE_RUN=old.REFERENCE_RUN;REFERENCE_COMMIT=old.REFERENCE_COMMIT;REFERENCE_RELEASE=old.REFERENCE_RELEASE
SEED=old.SEED;RX=FIT=old.RX;STEPS=old.STEPS;UPDATES=old.UPDATES
EPOCH_OFFSET=20;STEP_OFFSET=4440;CUMULATIVE_EPOCHS=30;CUMULATIVE_UPDATES=6660
read,write=old.read,old.write
PROJECT=old.PROJECT
for _key in ('SOURCE','FORMAL_VIEWS','TRUTH','PANEL','SOURCE_CACHE','VIEWS','SCENES','CLASSES','FULL_FP32_POLICY','TRAIN_SCENES','CHANNEL_SEEDS'):globals()[_key]=deepcopy(getattr(old,_key))
BASE=Path(PROJECT)/'runs'/RUN;ARMS=('B00','B01')
def rows():return old.rows()
def reference_config(row,segment=30):return old.reference_config(row,segment)
def config(row,segment=30):
 if row not in rows() or segment!=30:raise ValueError('Only two original P43 own20+10 rows')
 c=old.config(row,segment);c.pop('lraw')
 c.update(schema='FcrossRX_ordinary_channel78_v1',run_id=RUN,launch_owner=OWNER,method='last10_L_ordinary_AWGN_MP_bypass',output_root=str(BASE/row['row_id']/'source'),training_recipe='Original P43 six L forwards and all losses unchanged;only native L ordinary AWGN and MP execute then return their own input',benchmark_context='Original own P43 E20 student and ownEMA,not SAT18;first20 full ordinary retained;As is complete practical(raw),not ordinary',budget_authorization='Two original own20+10,totalE30/6660updates;P43 oldNoAs30/As30 frozen readouts reused;no new head/loss/NN',baseline_reference='Original P43 own20 initializers and originalNoAs30/As30 controls',cost_claim='6L768 plus original U/EMA unchanged;all AWGN/MP shadow computation retained;no compute saving')
 c['training_mechanism'].update(objective='Unchanged .5RDordinary128+.5RDmixed128+.25RDraw128+.1ORD128CROSS_RX',ordinary_return='original native chain with only MP/AWGN primitive outputs bypassed',U_recipe_unchanged=True)
 c['crossview']['actual_anchor_input']='native ordinary128 with MP/AWGN primitive outputs bypassed'
 c['ordinary_channel']=dict(scope='unique original L128 ordinary object only',removed=['AWGN','MP'],rule='Execute original primitive once on actual native input;return that primitive input;preserve all draws and simulated cost',first20_full_ordinary_retained=True,U_weak_strong_unchanged=True,U_augmentor_wrapped=False,As_raw_path_unchanged=True,other_native_operations_unchanged=True,post_MP_AWGN_sigma='native recalculated on bypassed MP input,total-chain effect',PA_input='native following bypassed general-channel prefix',primitive_patch_scope='one synchronous L ordinary call;finally restore',ordinary_calls_per_step=1,ordinary_packets_per_step=128,whole_training_RNG_matched=False,new_losses=0,new_parameters=0,new_NN=0,formal_test_changed=False)
 c['source_promotion']=dict(fullV_newAs_minus_newNoAs_clean_min_pp=-.1,panel_midlow_newAs_minus_originalAs_min_pp=.5,source_failure_skips_formal=False,automatic_extension=False,on_fail='close fixed joint bypass;no E50/newseed/single-component rescue')
 return c
def validate(c):
 row=next((r for r in rows() if r['row_id']==c.get('row_id')),None)
 if row is None or c!=config(row,c.get('segment')):raise ValueError('Fixed P78 scope/data/ownP43parent/budget differs')
 return c
def make_args(c,device='cuda:0'):
 validate(c);row=next(r for r in rows() if r['row_id']==c['row_id']);a=old.make_args(old.config(row),device)
 a.run_id=RUN;a.candidate_id=c['row_id'];a.output_dir=c['output_root'];a.baseline_ckpt=c['parent_source']+'/final_ssdg.pth'
 return a
