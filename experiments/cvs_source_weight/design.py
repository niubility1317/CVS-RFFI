"""Report route4 frozen L_s proxy; no U/target, optimizer or head fitting."""
from pathlib import Path
from experiments.cvs_moment_area50 import design as parent
ROOT=Path(__file__).resolve().parents[2]
PROJECT,SOURCE,FULL_FP32_POLICY,read,write=(getattr(parent,k) for k in ('PROJECT','SOURCE','FULL_FP32_POLICY','read','write'))
RUN='20261014-phase1-source-trusted-weight-manysig-s01-m1-r01';RELEASE='cvs_source_weight_20261014_r01';OWNER='codex/root/source-weight-20261014';BASE=Path(PROJECT)/'runs'/RUN
SEED=2026092701;ROW='SOURCE_WEIGHT-s2026092701';RX=(1,3,4,6,8);DAYS=(1,2,3)
PARAMETERS=tuple('id_backbone.encoder.core.id_backbone.'+n+'.pw.weight' for n in ('t1','t2','t3'))
def rows():return [dict(row_id=ROW,arm='SOURCE_WEIGHT',model_seed=SEED)]
def source_config():return parent.config(next(r for r in parent.rows() if r['arm']=='SIG_TIME'))
def config(row):
 if row!=rows()[0]:raise ValueError('Unregistered source weight row')
 return dict(schema='source_trusted_weight_proxy_v1',run_id=RUN,**row,source_config=source_config(),source_execution_commit='512c10a267b5e654cfd2c983ab1debdeebc7f626',output_root=(BASE/ROW).as_posix(),launch_owner=OWNER,source_role='L_s',source_pool_count=6300,source_sample_count=6300,batch=256,shuffle_seed=2026101411,dropout_seed=2026101412,teacher='own matching final50 EMA state; defaultcosine unchanged',student_mode='train only for fixed Dropout replay, no identity BN; frozen parameters except temporary feature derivative objects',hardness='1-student,strong probability of teacher pseudo class',weight='stopgrad(.75+.5*clamp(h/.1,0,1))',weight_bounds=[.75,1.25],hardness_scale=.1,loss='lambdaU.16*sum(old_accept*w*CE)/sum(old_accept*w); originalmaskunchanged',parameters=list(PARAMETERS),source_proxy_only=True,training_epochs=0,optimizer_updates=0,head_fit=False,query_read=False,U_IQ_read=False,U_truth_read=False,V_cache_materialized=True,V_forward=False,goal_claim=False,stop_rule='Role/provenance/nonfinite/shape/oldgate/modelstate/gradient or saved-recount failure preserves output; no retry or parameter sweep')
def validate(c):
 if c!=config(rows()[0]):raise ValueError('Frozen source weight config differs')
