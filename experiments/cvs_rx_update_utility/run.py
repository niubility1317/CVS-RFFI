from pathlib import Path
import os,sys,time,json
import torch
from torch.nn import functional as F
from . import design as d
from .utility import sample,gradient,direction,metrics,virtual,select_from_L
from experiments.cvs_rx_risk_learning import design as learning_design
from experiments.cvs_rx_risk_learning.run import load as load_original
from experiments.cvs_stage2_rx_risk.receiver import ReceiverActions,decision_state
from experiments.cvs_rx_risk_learning.learning import probabilities
from experiments.cvs_report_ssl.model import groups
from experiments.cvs_rx_intervention_audit.actions import apply
from experiments.cvs_rx_critic_audit.metrics import margin
from experiments.cvs_equivariant_identity.precision import numerical_context

def load(c,device):
 lc=learning_design.config(next(r for r in learning_design.rows() if r['model_seed']==c['model_seed']))
 model,unused,roles=load_original(lc,device);del unused
 pc=c['parent_build_config'];heads=ReceiverActions(pc,roles['L']).to(device).eval()
 pg,_=groups(model,pc);named={id(p):n for n,p in model.named_parameters()};params=[];scales=[]
 for group in pg:
  for p in group['params']:
   if named[id(p)].endswith(('low_hz_','band_hz_')):p.requires_grad_(False);continue
   params.append(p);scales.append(torch.full((p.numel(),),1. if group['name']=='encoder' else 3.,device=device))
 model.eval();return model,heads,roles,params,torch.cat(scales)

def batch(model,heads,roles,params,scales,c,b,device):
 L,V=roles['L'],roles['V'];rxs=sorted(L['rx'].unique().tolist());days=sorted(L['day'].unique().tolist());rx=rxs[b%5];day=days[(b//5)%3]
 gen=torch.Generator().manual_seed(c['model_seed']+9137+b*997)
 idx=sample(L,[rx],[day],c['budget']['update_per_class'],gen)
 other_rx=[r for r in rxs if r!=rx];other_day=[v for v in days if v!=day]
 li=sample(L,other_rx,other_day,c['budget']['probe_per_class_RX_day'],gen);vi=sample(V,other_rx,other_day,c['budget']['probe_per_class_RX_day'],gen)
 if torch.isin(idx,li).any():raise ValueError('Update/probe L overlap')
 probes={role:(cache['x'][i].to(device),cache['y'][i].to(device)) for role,cache,i in [('L',L,li),('V',V,vi)]}
 base={role:metrics(model,x,y) for role,(x,y) in probes.items()}
 x=L['x'][idx].to(device);y=L['y'][idx].to(device);before=[p.detach().clone() for p in params];norm=torch.cat([v.reshape(-1) for v in before]).norm()
 with torch.no_grad():
  candidates=heads.candidates(L,idx,True);state,z=decision_state(model,x);ns,nd=heads.normalized(state,candidates);risk_prob=probabilities(heads.critic(ns,nd),True)
 clean_ce=F.cross_entropy(model(x),y);gclean=gradient(clean_ce,params);losses=[];margins=[]
 for a in range(8):
  changed,_=apply(x,candidates[:,a],1.,33,3.);z=model(changed);losses.append(F.cross_entropy(z,y,reduction='none'));margins.append(margin(z.detach(),y))
 losses=torch.stack(losses,1);margins=torch.stack(margins,1);hard=margins.argmin(1)
 gactions=[.5*gclean+.5*gradient(losses[:,a].mean(),params) for a in range(8)]
 grads=dict(clean=gclean,uniform=.5*gclean+.5*gradient(losses.mean(),params),risk_weighted=.5*gclean+.5*gradient((losses*risk_prob).sum(1).mean(),params),hard_margin=.5*gclean+.5*gradient(losses.gather(1,hard[:,None]).mean(),params))
 # A transfer gradient uses L only; V remains read-only forward evaluation.
 gp=gradient(F.cross_entropy(model(probes['L'][0]),probes['L'][1]),params)
 result=dict(batch=b,update_RX=rx,update_day=day,update_ids=[L['ids'][i] for i in idx.tolist()],L_probe_ids=[L['ids'][i] for i in li.tolist()],V_probe_ids=[V['ids'][i] for i in vi.tolist()],baseline={k:v[0] for k,v in base.items()},candidate_mean_CE=losses.detach().mean(0).tolist(),candidate_mean_margin=margins.mean(0).tolist(),candidate_L_gradient_alignment=[float(gp.dot(direction(g,scales))) for g in gactions],risk_probability_mean=risk_prob.mean(0).tolist(),scales=[])
 for step in c['budget']['relative_steps']:
  delta=float(norm)*step
  candidate_results=[virtual(model,params,before,direction(g,scales),delta,probes,base) for g in gactions]
  chosen=select_from_L(candidate_results)
  policies={name:virtual(model,params,before,direction(g,scales),delta,probes,base) for name,g in grads.items()}
  policies['L_selected']=candidate_results[chosen]
  result['scales'].append(dict(relative_step=step,actual_delta_l2=delta,selected_from_L=chosen,candidates=candidate_results,policies=policies))
 if not all(torch.equal(p.detach(),v) for p,v in zip(params,before)):raise ValueError('Virtual update restore failed')
 return result

def run(row_id,device='cuda:0'):
 c=d.config(next(r for r in d.rows() if r['row_id']==row_id));d.validate(c);out=Path(c['output']);out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2);started=time.perf_counter()
 with numerical_context(d.parent.FULL_FP32_POLICY):
  model,heads,roles,params,scales=load(c,device);initial={k:v.detach().clone() for k,v in model.state_dict().items()};aux={k:v.detach().clone() for k,v in heads.state_dict().items()}
  d.write(out/'resolved_config.json',dict(config=c,pid=os.getpid(),cwd=os.getcwd(),python=sys.executable,commit=(d.ROOT/'release_commit.txt').read_text().strip(),CUDA_VISIBLE_DEVICES=os.environ.get('CUDA_VISIBLE_DEVICES'),U_file_opened=False,target_loaders_constructed=False,parameters=sum(p.numel() for p in params)))
  print('RESOLVED',json.dumps(c),flush=True)
  with (out/'utility_records.jsonl').open('x',encoding='utf-8') as stream:
   for b in range(c['budget']['batches']):
    r=batch(model,heads,roles,params,scales,c,b,device);stream.write(json.dumps(r,allow_nan=False)+'\n');stream.flush()
    if not all(torch.equal(initial[k],v) for k,v in model.state_dict().items()) or not all(torch.equal(aux[k],v) for k,v in heads.state_dict().items()):raise ValueError('Frozen state changed')
    print('BATCH',b,json.dumps([dict(relative_step=x['relative_step'],chosen=x['selected_from_L'],V_CE_change={k:v['V']['CE_change'] for k,v in x['policies'].items()}) for x in r['scales']]),flush=True)
  d.write(out/'completion.json',dict(status='SOURCE_DIAGNOSTIC_COMPLETE',config=c,identity_updates=0,auxiliary_updates=0,target_access=False,V_gradients=False,U_file_opened=False,batches=30,virtual_steps=30*2*12,source_model_unchanged=True,auxiliary_unchanged=True,elapsed_seconds=time.perf_counter()-started,hardware=torch.cuda.get_device_name() if device.startswith('cuda') else 'CPU',peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated() if device.startswith('cuda') else None))
