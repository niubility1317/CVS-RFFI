"""Closed-form metadata/CM/cost controls;zero real IQ/CP/NN/target."""
import copy,gzip,json,tempfile
from pathlib import Path
from analysis import cvs_LSAM_recount_20261018 as a
def write(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x),encoding='utf8')
def run():
 negative=0
 def bad(f):
  nonlocal negative
  try:f()
  except ValueError:negative+=1
  else:raise AssertionError('Invalid source summary accepted')
 with tempfile.TemporaryDirectory(prefix='LSAM_recount_') as t:
  project=Path(t);B=project/'runs'/a.RUN;meta=dict(ids=[],TX=[],RX=[],day=[])
  for y in range(6):
   for rx in (1,3,4,6,8):
    for day in (1,2,3):
     for n in range(20):meta['ids'].append(str(len(meta['ids'])));meta['TX'].append(y);meta['RX'].append(rx);meta['day'].append(day)
  for mode in a.MODES:
   P=B/(mode+'-s2026092701')/'source_qualification';records=[];metrics={}
   for view in a.VIEWS:
    n=1 if view=='clean' else 2;metrics[view]=dict(accuracy=1.,decisions=1800*n)
    for draw in range(n):records.append(dict(view=view,draw=draw,count=1800,correct=1800,confusion=[[300 if i==j else 0 for j in range(6)] for i in range(6)],per_RX={str(r):dict(count=360,correct=360) for r in (1,3,4,6,8)}))
   r=dict(status='SOURCE_QUALIFIED_FROZEN_READOUT_ONLY',source_V_unique=1800,source_cells=90,all_RX=[1,3,4,6,8],source_RX_holdout=None,CP=1,NN=23400,physical_channel_calls=21600,updates=0,teacher_NN=0,head_fit_updates=0,U_truth_read=False,target=False,model_state_unchanged=True,metrics=metrics,records=records,cache_loads=1,wall_seconds=.1);write(P/'result.json',r);write(P/'source_ids.json',meta)
  decision=dict(conditions=dict(clean=True,midurban=True,lowurban=False,suburban=True),delta_pp={v:0. for v in a.VIEWS},source_pass=False,target_access=False,decision='CLOSED_FIXED_LSAM_SOURCE_RECOGNITION_FAILED');write(B/'source_matrix_decision.json',decision);verified=a.source_qualification(project);assert verified['CM_recounts']==39 and verified['source_pass'] is False and verified['rows']['AS_LSAM']['metrics']['clean']['macro_F1']==1.
  P=B/'AS_LSAM-s2026092701/source_qualification';original=json.loads((P/'result.json').read_text());x=copy.deepcopy(original);x['records'][0]['per_RX']['6']['count']=359;write(P/'result.json',x);bad(lambda:a.source_qualification(project))
  x=copy.deepcopy(original);x['target']=True;write(P/'result.json',x);bad(lambda:a.source_qualification(project))
  x=copy.deepcopy(original);x['records'].pop();write(P/'result.json',x);bad(lambda:a.source_qualification(project))
  write(P/'result.json',original);m=copy.deepcopy(meta);m['TX'][0],m['TX'][300]=m['TX'][300],m['TX'][0];write(P/'source_ids.json',m);bad(lambda:a.source_qualification(project));write(P/'source_ids.json',meta)
  S=project/'synthetic_source';S.mkdir();r=dict(mode='REPLAY',radius=0.,step=0,cumulative_step=0,L_forwards=8,extra_L_forward_packets=512,extra_VJP_calls=2,U_replayed=False,teacher_replayed=False,weights_exactly_restored=True,global_RNG_restored=True,persistent_buffers_unchanged=True,L_loss=1.,replay_L_loss=1.,L_gradient_norm=2.,replay_gradient_norm=2.,replay_gradient_relative_difference=0.,actual_perturbation_L2=0.,host_enqueue_seconds=.01)
  data=[dict(r,step=i,cumulative_step=i+4440) for i in range(1,2221)];cost=dict(update_mode='REPLAY',extra_supervised_VJP_calls=4440,replay_calls=2220,replay_L_forward_packets=1136640,base_L_forward_packets=1136640,L_forward_packets=2273280,perturbed_calls=0,student_train_forward_packets=2840280,teacher_train_forward_packets=0,teacher_eval_forward_packets=567000)
  def save(rows):
   with gzip.open(S/'L_replay_steps.jsonl.gz','wt',encoding='utf8') as f:
    for x in rows:f.write(json.dumps(x)+'\n')
  save(data);v=a.replay_records(S,'AS_REPLAY',cost);assert v['total_L_NN']==2273280 and v['extra_L_backward_packet_derivatives']==2273280 and len(v['epochs'])==10
  changed=copy.deepcopy(data);changed[0]['U_replayed']=True;save(changed);bad(lambda:a.replay_records(S,'AS_REPLAY',cost));save(data[:-1]);bad(lambda:a.replay_records(S,'AS_REPLAY',cost));save(data);bad(lambda:a.replay_records(S,'AS_REPLAY',dict(cost,L_forward_packets=1136640)))
  changed=copy.deepcopy(data);changed[7]['replay_gradient_relative_difference']=.01;save(changed);bad(lambda:a.replay_records(S,'AS_REPLAY',cost))
 return dict(status='PASS',source_CM_fixture=39,source_decision='correctlyFAIL_zeroIncrement',replay_records_fixture=2220,protocol_negatives=negative,IQ=0,CP=0,NN=0,updates=0,target=False)
if __name__=='__main__':print(json.dumps(run()))
