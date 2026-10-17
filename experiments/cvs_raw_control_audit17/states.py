"""Local CPU comparison of already-authorized random init tensors;0NN."""
import argparse,json,os,sys,time
from itertools import combinations
from pathlib import Path
import torch
from . import design as d
HEAD='id_backbone.encoder.core.id_backbone.cls_head.weight'
def checked(state):
 if not isinstance(state,dict) or len(state)!=206 or HEAD not in state or tuple(state[HEAD].shape)!=(6,160) or any(not torch.is_tensor(v) or not bool(torch.isfinite(v).all()) for v in state.values()):raise ValueError('Finite original206state/defaulthead6x160 required')
 return state
def compare(a,b):
 checked(a);checked(b)
 if set(a)!=set(b):return dict(keys_equal=False,tensors_equal=False,changed_names=sorted(set(a)^set(b)))
 changed=[];maxdiff=0.;sq=0.;shapes=True
 for n in a:
  x,y=a[n],b[n]
  if x.shape!=y.shape or x.dtype!=y.dtype:shapes=False;changed.append(n);continue
  if not torch.equal(x,y):changed.append(n);v=(x.double()-y.double());maxdiff=max(maxdiff,float(v.abs().max()));sq+=float(v.square().sum())
 return dict(keys_equal=True,shapes_dtypes_equal=shapes,tensors_equal=not changed,changed_names=changed,max_absolute_difference=maxdiff,L2_difference=sq**.5)
def run(inp,out):
 inp,out=Path(inp),Path(out)
 if out.exists():raise FileExistsError('Reconcile existing immutable local analysis')
 configs=[d.verify_source(inp/r['key'],r) for r in d.sources()]
 if any(v['contract']['role_ids']!=configs[0]['contract']['role_ids'] for v in configs[1:]):raise ValueError('Exact matching effective source role IDs before initialstate reads')
 out.mkdir();torch.set_num_threads(2);start=time.perf_counter();states={};first=None;comparisons={}
 d.write(out/'execution.json',dict(pid=os.getpid(),cwd=os.getcwd(),argv=sys.argv,python=sys.executable,torch=torch.__version__,CPU_threads=2,NN=0,updates=0,source_only=True,target_access=False,read_at=time.time()))
 for r in d.sources():
  p=inp/r['key']/'initial_smoke.pt';payload=torch.load(p,map_location='cpu',weights_only=False);s=checked(payload['model'])
  if first is None:
   first=s;d.write(out/'real_initialCP_noquery_smoke.json',dict(status='PASS',strict_structure=True,state_tensors=206,head_shape=[6,160],checkpoint_read=1,IQ=0,NN=0,updates=0,target_access=False))
  comparisons[r['key']]=compare(first,s);states[r['key']]=s;del payload
 pairwise={a+'|'+b:compare(states[a],states[b]) for a,b in combinations(states,2)}
 result=dict(status='VERIFIED',pairwise=pairwise,first_reference=d.sources()[0]['key'],comparisons=comparisons,all_five_initialstates_equal=all(v['tensors_equal'] for v in comparisons.values()),initial_checkpoint_loads=5,trained_checkpoint_loads=0,IQ_reads=0,network_forwards=0,optimizer_updates=0,head_fit_updates=0,target_access=False,wall_seconds=time.perf_counter()-start)
 d.write(out/'initial_states.json',result);print(json.dumps(result))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--output',required=True);a=p.parse_args();run(a.input,a.output)
