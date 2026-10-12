"""Read-only fixed-source endpoint diagnostic; never reads target or selects a row.

All 21 rows, fixed first private-domain-stream batch, no optimizer or updates.
Endpoint cosines do not reconstruct missing during-training cosine history.
"""
import argparse,json,sys,time
from pathlib import Path

def module_key(name):
 if '.core.id_backbone.' in name:return 'core.'+name.split('.core.id_backbone.',1)[1].split('.')[0]
 if '.encoder.' in name:return 'response.'+name.split('.encoder.',1)[1].split('.')[0]
 return name.rsplit('.',1)[0]

def alignment(gradients,groups,torch):
 result={}
 for group,indices in groups.items():
  g={task:[gradients[task][i] for i in indices] for task in gradients}
  norms={task:float(sum((t.square().sum() for t in values if t is not None),torch.zeros(())).sqrt()) for task,values in g.items()}
  cos={}
  for a,b in [('ce','KL'),('ce','extra'),('KL','extra')]:
   dot=float(sum((x.mul(y).sum() for x,y in zip(g[a],g[b]) if x is not None and y is not None),torch.zeros(())))
   denominator=norms[a]*norms[b];cos[a+':'+b]=dot/denominator if denominator>1e-12 else None
  result[group]=dict(weighted_gradient_norms=norms,cosines=cos)
 return result

def run(release):
 sys.path.insert(0,str(release));import torch
 from experiments.cvs_e20_response import design as d
 from experiments.cvs_e20_response.source import load_source
 from experiments.cvs_e20_response.model import build,groups
 from experiments.cvs_e20_response.objective import objective
 from experiments.cvs_e20_response.response import Bank
 from experiments.cvs_e20_response.evaluate import snapshot,source_provenance
 from experiments.cvs_nocfo_engineering.peer_cache import load as load_peer
 from experiments.cvs_nocfo_engineering.teacher import verify
 from experiments.cvs_report_ssl.sampling import DomainStream
 from experiments.cvs_stage2_decouple.source import seed_all
 from experiments.cvs_equivariant_identity.precision import numerical_context
 torch.set_num_threads(2);configs=snapshot();rows=[];start=time.perf_counter()
 with numerical_context(d.FULL_FP32_POLICY):
  for c in configs:
   source=source_provenance(c);L,U,V,contract=load_source(c);seed_all(c['model_seed']);model=build(c,'cpu',source/'final_ssdg.pth');model.train();pg,names=groups(model,c)
   peer=load_peer(c,U) if c['unlabeled'] else None;bank=Bank(c,L,U,'cpu') if c['response']['enabled'] else None
   li,_=DomainStream(L['rx'],L['day'],c['model_seed']+101).batch(128);ui=DomainStream(U['rx'],U['day'],c['model_seed']+901337).batch(128)[0] if U is not None else None
   before={n:t.clone() for n,t in model.state_dict().items()};fixed={k:{n:t.clone() for n,t in reader.state_dict().items()} for k,reader in bank.readers.items()} if bank else None
   loss,parts,weighted,details=objective(c,model,peer,bank,L,li,U,ui,4440,'cpu');parameters=dict(model.named_parameters());owned={id(p) for g in pg for p in g['params']};selected=[(n,p) for n,p in parameters.items() if id(p) in owned and not n.endswith(('low_hz_','band_hz_'))]
   gradients={k:torch.autograd.grad(v,[p for n,p in selected],retain_graph=True,allow_unused=True) for k,v in weighted.items()}
   buckets={g['name']:[i for i,(n,p) in enumerate(selected) if id(p) in {id(x) for x in g['params']}] for g in pg}
   for i,(name,p) in enumerate(selected):buckets.setdefault('module:'+module_key(name),[]).append(i)
   audit=alignment(gradients,buckets,torch)
   if not torch.isfinite(loss) or any(not torch.equal(t,model.state_dict()[n]) for n,t in before.items()) or any(p.grad is not None for p in model.parameters()):raise ValueError('Read-only endpoint changed identity state')
   if bank and any(p.grad is not None for reader in bank.readers.values() for p in reader.parameters()):raise ValueError('Endpoint changed fixed reader gradients')
   if bank and any(not torch.equal(t,fixed[key][n]) for key,reader in bank.readers.items() for n,t in reader.state_dict().items()):raise ValueError('Endpoint changed fixed reader state')
   parent=verify(c['model_seed']);state=torch.load(parent['checkpoint'],map_location='cpu',weights_only=False)['model'];movement={}
   for name,p in parameters.items():
    if name not in state or state[name].shape!=p.shape:raise ValueError('Parent state coordinates differ')
    key=module_key(name);entry=movement.setdefault(key,dict(delta_squared=0.,parent_squared=0.,parameters=0));entry['delta_squared']+=float((p.detach()-state[name]).square().sum());entry['parent_squared']+=float(state[name].square().sum());entry['parameters']+=p.numel()
   for entry in movement.values():entry['delta_l2']=entry.pop('delta_squared')**.5;entry['parent_l2']=entry.pop('parent_squared')**.5;entry['relative_l2']=entry['delta_l2']/max(entry['parent_l2'],1e-10)
   rows.append(dict(row_id=c['row_id'],arm=c['arm'],model_seed=c['model_seed'],phase='fixed E40 endpoint only',weighted_losses={k:float(v.detach()) for k,v in weighted.items()},audit=audit,module_parameter_movement=movement,source_L_batch=128,source_U_batch=128 if U is not None else 0,source_L_PIDs=[L['ids'][i] for i in li.tolist()],source_U_PIDs=[U['ids'][i] for i in ui.tolist()] if U is not None else [],forward_counts=details,identity_updates=0,auxiliary_updates=0,target_access=False,V_forward=False,U_truth=False,state_unchanged=True))
 if len(rows)!=21:raise ValueError('Missing endpoint row')
 return dict(status='VERIFIED',run_id=d.RUN,source_only=True,all_rows=21,rows=rows,identity_updates=0,auxiliary_updates=0,target_access=False,training_cosine_history_available=False,training_module_history_scope='encoder/fusion_head only; endpoint finer native modules calculated from own E40 versus corresponding E20',frequency_gradients_excluded=True,sampling='fixed first128 private L/U DomainStream; one endpoint batch, not full-source or temporal causal analysis',seconds=time.perf_counter()-start)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--release',type=Path,required=True);a=p.parse_args();print(json.dumps(run(a.release),allow_nan=False))
