import argparse,json,os,sys,time
from pathlib import Path
import torch
from torch.nn import functional as F
from . import design as d
from .geometry import terms,record,aggregate,gradient_vectors,gradient_summary
from experiments.cvs_stage2_rx_risk.source import load_source
from experiments.cvs_stage2_rx_risk.model import build,groups
from experiments.cvs_stage2_rx_actions.receiver import ReceiverActions
from experiments.cvs_report_ssl.sampling import DomainStream
from experiments.cvs_equivariant_identity.precision import numerical_context,actual_flags

def load(c,device):
 pc=c['parent_build_config'];L,U,_,contract,provenance=load_source(pc)
 model=build(pc,device).eval();pg,_=groups(model,pc)
 heads=ReceiverActions(pc,L).to(device).eval()
 params=[p for n,p in model.named_parameters() if p.requires_grad and not n.endswith(('low_hz_','band_hz_'))]
 return model,heads,L,U,params,contract,provenance

@torch.no_grad()
def view(c,heads,cache,indices,labeled,kind,generator,device):
 x=cache['x'][indices].to(device)
 if kind=='physical_noise':x,_,_=heads.choose(x,cache,indices,labeled,1000,generator,uniform=True)
 elif kind!='noise':raise ValueError('Unknown view')
 return x+torch.randn(x.shape,generator=generator,dtype=x.dtype).to(device)*c['noise_std']

def forward(model,x):
 out=model.forward_identity_only(x)
 return out['tx_logits'],out['id_feat_joint']

@torch.no_grad()
def full_role(c,model,heads,cache,labeled,kind,device):
 generator=torch.Generator().manual_seed(c['model_seed']+701+(0 if labeled else 100))
 collected={};n=len(cache['x']);rank=None
 for start in range(0,n,c['batch_size']):
  idx=torch.arange(start,min(n,start+c['batch_size']));x=cache['x'][idx].to(device)
  base,bf=forward(model,x);changed,z=forward(model,view(c,heads,cache,idx,labeled,kind,generator,device))
  values,rank=record(changed,z,base,bf,model.head.weight,cache['y'][idx].to(device) if labeled else None)
  for k,v in values.items():collected.setdefault(k,[]).append(v.cpu())
 values={k:torch.cat(v) for k,v in collected.items()}
 result=aggregate(values,cache['rx'],cache['day'],c['confidence_bins'])
 return dict(count=n,head_centered_rank=rank,strata=result,U_truth_access=False)

def gradients(c,model,heads,L,U,params,device,count=None):
 ls=DomainStream(L['rx'],L['day'],c['model_seed']+101);us=DomainStream(U['rx'],U['day'],c['model_seed']+901337)
 gen=torch.Generator().manual_seed(c['model_seed']+778);rows=[]
 for step in range(count or c['gradient_batches']):
  li,_=ls.batch(c['gradient_batch_size']);ui,_=us.batch(c['gradient_batch_size'])
  # L is only a reference gradient. U labels are unavailable and never inspected.
  x=L['x'][li].to(device);y=L['y'][li].to(device);ux=U['x'][ui].to(device)
  with torch.no_grad():base,bf=forward(model,ux)
  for kind in c['views']:
   ce=F.cross_entropy(model(x),y);changed,z=forward(model,view(c,heads,U,ui,False,kind,gen,device))
   losses={'L_CE':ce,**{k:v.mean() for k,v in terms(changed,z,base,bf).items()}}
   vectors=gradient_vectors(losses,params)
   rows.append(dict(step=step,view=kind,accepted_U=int((base.softmax(1).max(1).values>=.95).sum()),
    losses={k:float(v.detach()) for k,v in losses.items()},gradients=gradient_summary(vectors)))
 return rows

def run(row_id,device='cuda:0'):
 c=d.config(next(r for r in d.rows() if r['row_id']==row_id));d.validate(c)
 out=Path(c['output']);out.mkdir(parents=True,exist_ok=False);started=time.perf_counter();torch.set_num_threads(2)
 d.write(out/'resolved_config.json',dict(config=c,pid=os.getpid(),cwd=os.getcwd(),python=sys.executable,
  commit=(d.ROOT/'release_commit.txt').read_text().strip(),CUDA_VISIBLE_DEVICES=os.environ.get('CUDA_VISIBLE_DEVICES'),target_loaders_constructed=False))
 with numerical_context(d.parent.FULL_FP32_POLICY):
  model,heads,L,U,params,contract,provenance=load(c,device)
  before={k:v.detach().clone() for k,v in model.state_dict().items()}
  bank={k:v.detach().clone() for k,v in heads.state_dict().items()}
  if str(device).startswith('cuda'):torch.cuda.reset_peak_memory_stats()
  d.write(out/'provenance.json',dict(original_E200=provenance,source_contract=contract,V_used=False,U_truth_access=False))
  results={}
  for role,cache,labeled in [('L',L,True),('U',U,False)]:
   for kind in c['views']:
    key=role+'_'+kind;results[key]=full_role(c,model,heads,cache,labeled,kind,device)
    d.write(out/'full_source_geometry.json',results)
    print('FULL_SOURCE_GEOMETRY '+json.dumps(dict(key=key,**results[key]['strata']['all'])),flush=True)
  grad=gradients(c,model,heads,L,U,params,device);d.write(out/'gradient_probes.json',grad)
  if not all(torch.equal(before[k],v) for k,v in model.state_dict().items()):raise ValueError('Identity state changed')
  if not all(torch.equal(bank[k],v) for k,v in heads.state_dict().items()):raise ValueError('Action bank changed')
  done=dict(status='SOURCE_DIAGNOSTIC_COMPLETE',row_id=row_id,identity_updates=0,target_access=False,V_used=False,U_truth_access=False,
   source_model_unchanged=True,bank_unchanged=True,roles=dict(L=6300,U=56700),views=2,gradient_records=len(grad),
   elapsed_seconds=time.perf_counter()-started,peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated() if str(device).startswith('cuda') else None,backend=actual_flags())
  d.write(out/'completion.json',done);print(json.dumps(done),flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--row',required=True);a=p.parse_args();run(a.row)
