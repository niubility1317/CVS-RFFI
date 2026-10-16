import json
from pathlib import Path
from experiments.cvs_f_crossrx_practical20 import design as prior
ROOT=Path(__file__).resolve().parents[2]
PROJECT,SOURCE,FULL_FP32_POLICY,SEED=(getattr(prior,k) for k in ('PROJECT','SOURCE','FULL_FP32_POLICY','SEED'))
RUN='20261016-diagnostic-FcrossRX-cleanAs-gradients-sourcefit-s2026092701-m2-r01'
RELEASE='cvs_FcrossRX_cleanAs_gradients_sourcefit_20261016_r01'
OWNER='codex/root/FcrossRX-cleanAs-gradients-20261016'
BASE=Path(PROJECT)/'runs'/RUN
PREPARATION=Path(PROJECT)/'runs/20261011-phase1-stage2-geometry-source-manysig-m3-r01/s2026092701'
FIT=(3,4,6,8);REPS=(0,1);BATCH=128
SEEDS=dict(order=20261017201,ordinary=20261017202,dropout_primary=20261017203,dropout_secondary=20261017204,channel=20261017205,receiver=20261016423)
REFERENCES={a:Path(PROJECT)/'runs'/prior.RUN/f'{a}-RX1-s{SEED}'/'source' for a in ('C0','C1')}
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def rows():return [dict(row_id='frozen-C0-C1-fitL5040',arm='gradient_interference')]
def config(row):
 if row not in rows():raise ValueError('One fixed frozen-source diagnostic row')
 return dict(schema='FcrossRX_cleanAs_gradient_diagnostic_v1',run_id=RUN,launch_owner=OWNER,**row,output_root=(BASE/row['row_id']).as_posix(),
  source_contract=SOURCE,source_preparation=PREPARATION.as_posix(),references={k:v.as_posix() for k,v in REFERENCES.items()},model_seed=SEED,
  source_RX=list(FIT),held_RX1_used=False,source_L_count=5040,source_TX=list(range(6)),source_days=[1,2,3],batch_size=BATCH,replicates=list(REPS),seeds=SEEDS,
  network='Unchanged F+CROSS_RX/NoExplicitCFO/joint160/defaultcos30; two existing exact current source-fold E20 checkpoints',
  ordinary='Actual shared native base config and E13-capped configure_augmentor_for_epoch at E20,one draw per batch/rep; reused across models',
  channel='Original full practical residual/post_sync/EQoff G1/25MHz/reference30/reflect; nominal3 training scenes only,one draw per L/rep; no state rejection',
  losses='Native RD(alpha1,LS.01),original shallow/mid SupCon(.1,first clean pair only); all visible L use true TX; no added training loss or optimizer',
  gradients='Primary=.5*d(RD_ordinary)+.1*d(SupCon); secondary=.5*d(RD_sameordinary) control or .5*d(RD_As). PhysicalHz Sinc gradient=g_q/25000. Group and whole dot-products.',
  stochastic_control='same primary/dropout-secondary streams across C0,C1 and secondary control/As; order independent of labels; all5040 included,final48 retained',
  diagnostic_boundary='FitL only; gradient dot products are Euclidean first-order geometry,not actual AdamW motion or unseen-domain accuracy; no positive result licenses promotion',
  intervention_rule='Only consider separately registered training-organization experiment if BOTH checkpoints and BOTH replicate streams have median full-model As/primary norm ratio>1 and As primary-opposed fraction >= matched-control fraction+.20. Otherwise abandon gradient-dominance rationale; no group/RX/rep/threshold rescue.',
  frozen_parameters=True,no_BatchNorm_required=True,state_unchanged_required=True,U_values_used=False,V_values_used=False,target_access=False,
  checkpoint_loads=2,new_trained_models=0,model_epochs=0,optimizer_updates=0,head_fit_updates=0,EMA_updates=0,
  expected_model_packets=120964,expected_decisions=60480,source_only=True,automatic_training=False,goal_completed=False)
def validate(c):
 if c!=config(rows()[0]):raise ValueError('Fixed source gradient design differs')
def verify_reference(arm,done,init,contract,freeze):
 expected=prior.config(prior.rows()[('C0','C1').index(arm)])
 if done['config']!=expected or done['epoch']!=20 or done['optimizer_steps']!=4440 or done['initialization']!='scratch' or done['checkpoint_sources'] or done['ancestors'] or done['head_fit_updates'] or done['target_access'] or not init['scratch_only'] or init['checkpoint_sources'] or init['target_contact'] or freeze['epoch']!=20 or freeze['selection']!='final_only' or freeze['target_access'] or contract['held_RX']!=1 or contract['source_receivers']!=list(FIT) or {k:len(v) for k,v in contract['role_ids'].items()}!={'L_s':5040,'U_s':45360,'V':21600}:raise ValueError('Exact existing current C0/C1 scratch E20 source provenance required')
 return contract
