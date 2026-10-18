"""Reuse complete fixed final default-head sourceV27000 readout;no extra NN."""
from . import design as d

def run(c):
 done=d.read(d.BASE/c['row_id']/'source/completion.json');out=d.BASE/c['row_id']/'source_qualification';out.mkdir(exist_ok=False);v=done['final_source_metrics']
 if done['status']!='SOURCE_TRAINED' or done['config']!=c or done['target_access'] or done['head_fit_updates'] or v['source_val_count']!=27000 or set(v['source_val_rx_accuracy'])!=set(map(str,d.FIT)):raise ValueError('Complete sourcefinal27000/originalhead only')
 result=dict(status='SOURCE_QUALIFIED_FROZEN_READOUT_ONLY',row_id=c['row_id'],metrics=v,source_V_unique=27000,source_cells=90,all_RX=list(d.FIT),source_RX_holdout=None,IQ_points=256,CP=0,NN=0,teacher_NN=0,physical_channel_calls=0,updates=0,head_fit_updates=0,U_truth_read=False,target=False,readout_reused_from='native finalsourceevaluation/each originalV once',model_state_unchanged=True)
 d.write(out/'result.json',result);print('SOURCE_QUALIFICATION_COMPLETE '+c['row_id'],flush=True);return result

def matrix():
 data={r['arm']:d.read(d.BASE/r['row_id']/'source_qualification/result.json') for r in d.rows()}
 if any(v['source_V_unique']!=27000 or v['all_RX']!=list(d.FIT) or v['target'] or not v['model_state_unchanged'] for v in data.values()):raise ValueError('Original fullsourceV readout required')
 a,b=(data[k]['metrics'] for k in ('B10','B00'));delta=(round(a['source_val_accuracy']*27000)-round(b['source_val_accuracy']*27000))*100/27000
 rx={str(r):(round(a['source_val_rx_accuracy'][str(r)]*5400)-round(b['source_val_rx_accuracy'][str(r)]*5400))*100/5400 for r in d.FIT};checks=dict(fullV_clean_gain=delta>=.5-1e-12,each_RX_protected=all(v>=-.25-1e-12 for v in rx.values()))
 passed=all(checks.values());result=dict(status='SOURCE_MATRIX_COMPLETE',source_pass=passed,decision='SOURCE_REPEAT_SHRINK_INDEPENDENT_VALUE_QUALIFIED_ONLY' if passed else 'CLOSED_FIXED_REPEAT_SHRINK_SOURCE_GAIN_FAILED',conditions=checks,clean_delta_pp=delta,RX_delta_pp=rx,rows=data,target_access=False,decision_before_primary_formal_metric_read=True,joint_goal_complete=False,new_Ag=True,Ag_As_interaction_measured=False)
 d.write(d.BASE/'source_matrix_decision.json',result);return result
