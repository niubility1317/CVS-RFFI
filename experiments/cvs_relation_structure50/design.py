from pathlib import Path
from experiments.cvs_evidence_relation import design as relation
from experiments.cvs_time_factorial50 import design as temporal
ROOT=Path(__file__).resolve().parents[2]
PROJECT,SOURCE,CAPSULE,TRUTH,CLASSES,SEEDS,FULL_FP32_POLICY,read,write=(getattr(relation,k) for k in ('PROJECT','SOURCE','CAPSULE','TRUTH','CLASSES','SEEDS','FULL_FP32_POLICY','read','write'))
RUN='20261014-phase1-relation-temporal-factorial-oldgate-e50-clean-manysig-m12-r01'
RELEASE='cvs_relation_structure50_20261014_r01';OWNER='codex/root/relation-structure-20261014';BASE=Path(PROJECT)/'runs'/RUN
ARMS=('COMPACT_ZERO','COMPACT_RESP','LONG_ZERO','LONG_RESP');PRIMARY=('LONG_RESP',);VIEWS=('clean',);READOUTS=('NATIVE_HEAD',)
TITLE='已验证时序结构与关系响应机制：独立匹配组合'
PAIRS=[('COMPACT_RESP','COMPACT_ZERO'),('LONG_RESP','LONG_ZERO'),('LONG_ZERO','COMPACT_ZERO'),('LONG_RESP','COMPACT_RESP')]
SOURCE_GATE=dict(final_L_CE_max=.5,V_mean_drop_max=.01,V_worst_RX_drop_max=.02)

def rows():return [dict(row_id=f'{a}-s{s}',arm=a,model_seed=s) for s in SEEDS for a in ARMS]
def baseline_config(seed):return relation.baseline_config(seed)
def temporal_config(row):
 a='LONG_SILU' if row['arm'].startswith('LONG') else 'COMPACT_LINEAR'
 return temporal.config(next(r for r in temporal.rows() if r['arm']==a and r['model_seed']==row['model_seed']))
def config(row):
 if row not in rows():raise ValueError('Unregistered relation/structure row')
 mode='CROSS_RESP' if row['arm'].endswith('RESP') else 'ZERO'
 rc=relation.config(next(r for r in relation.rows() if r['arm']==mode and r['model_seed']==row['model_seed']))
 tc=temporal_config(row);rel=dict(rc['relation'],mode=mode);features=dict(tc['feature_extractor'],structural_changes=True,mechanism_extra_parameters=0,training_mechanism='unchanged crossRX relation and signed PA/ref response')
 return dict(rc,schema='relation_structure50_v1',run_id=RUN,**row,launch_owner=OWNER,output_root=(BASE/row['row_id']/'source').as_posix(),feature_extractor=features,relation=rel,structure_factor='LONG_SILU' if row['arm'].startswith('LONG') else 'COMPACT_LINEAR',mechanism_factor='CROSS_RESP' if row['arm'].endswith('RESP') else 'ZERO',structure_origin=dict(run_id=temporal.RUN,initialization='same original48213 CPU fork_rng; all four new tensor draws identical; compact/long settings differ',checkpoint_inherited=False),source_falsification=dict(SOURCE_GATE,nonzero_relation_fraction_min=.5,mean_aux_to_L_feature_gradient_min=1e-5),goal=dict(primary='LONG_RESP',clean_three_seed_mean=.84,feedback_to_training=False),cost_claim='12new scratch50/133200updates, first4seed01 only then fixed source expansion; no historical target decisions reused,3060anchor Lforward/model; no headfit',combination_authorization='User asks prior clear positive routes continue. Existing goal preregisters matched combinations after CROSS_RESP source pass; original firstseed source passed before target read. No new loss parameter, temperature, schedule or target-based parent selection.')
def validate(c):
 row=next((r for r in rows() if r['row_id']==c.get('row_id')),None)
 if row is None or c!=config(row):raise ValueError('Fixed relation/structure factorial differs')
 return c
def make_args(c,device='cuda:0'):
 validate(c);a=relation.old.make_args(baseline_config(c['model_seed']),device);a.output_dir=c['output_root'];a.run_id=RUN;a.candidate_id=c['row_id'];return a
def result_metadata():return dict(rows=12,new_backbones=12,existing_backbone_references=0,readouts=12,prediction_count=2016000,reused_native_decisions=0,new_prediction_decisions=2016000,metric_records=216,backbone_updates_new=133200,historical_baseline_updates=0,factorial=True)
