from pathlib import Path
from . import design as d
from .model import installed
from experiments.cvs_pa_cross_delay.source import train as shared_train,source_validation
from experiments.cvs_default_head.source_health import preserving_rng

def details(model):return dict(identity_input_policy=model._identity_input_policy,actual_identity_mixstyle=model._source_mixstyle)
def epoch_diagnostic(c,state,row):
 b=state['model'].id_backbone;epoch=int(row['epoch']);records=b._source_style_history;expected=222 if c['identity_mixstyle']['enabled'] else 0
 if len(records)!=expected or (expected and [r['call'] for r in records]!=list(range((epoch-1)*222+1,epoch*222+1))):raise ValueError('Exactly one main L hook call per actual update required')
 layer_counts={}
 for layer in d.STYLE_LAYERS:
  items=[r['layers'][layer] for r in records]
  layer_counts[layer]=dict(calls=len(items),enabled_calls=sum(v['enabled'] for v in items),packet_calls=sum(r['count'] for r in records),eligible_packets=sum(v['eligible_packets'] for v in items),changed_packets=sum(v['changed_packets'] for v in items))
 record=dict(epoch=epoch,main_L_calls=len(records),layers=layer_counts,teacher_style_calls=0,U_style_calls=0,eval_style_calls=0)
 if epoch in (33,34):record['source_V_RX_health']=preserving_rng(state['model'],state['ctx']['val_loader'],'cuda:0',source_validation)
 state.setdefault('style_diagnostics',[]).append(record);d.write(Path(c['output_root'])/'source_style_epoch_diagnostic.json',dict(records=state['style_diagnostics'],source_only=True,U_truth_read=False,head_fit=False,query_read=False,extra_V_passes=sum('source_V_RX_health' in r for r in state['style_diagnostics'])))
 records.clear()

# Run-owned source witness before worker proceeds to fixed clean prediction.
def train(c):
 shared_train(c,design=d,installer=installed,details=details,epoch_callback=epoch_diagnostic)
 from .feature_check import run
 proof=run(c['model_seed']);own=next((r for r in proof['rows'] if r['row_id']==c['row_id']),None)
 if own is None:raise ValueError('Actual source feature witness missing')
 d.write(Path(c['output_root'])/'source_feature_witness.json',dict(row=own,source_only=True,query_read=False,head_fit=False,V_cache_materialized=True,V_forward=False))
