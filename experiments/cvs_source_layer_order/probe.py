import json,time,traceback
from pathlib import Path
import torch
from torch.nn import functional as F
from . import design as d
from .objective import capture,views,relation_indices,losses,metadata
from experiments.cvs_signature_stats50.predict import source_proof,own_model
from experiments.cvs_trusted_weight50 import design as pd,model as pm
from experiments.cvs_equivariant_identity.precision import numerical_context

def model():return own_model(d.source_config(),'cpu',design=pd,build_fn=pm.build)

def cohorts(L):
 metadata(L['y'],L['rx'],L['day'],L['ids'])
 if len(L['ids'])!=6300:raise ValueError('Expected existing L6300')
 order=torch.randperm(6300,generator=torch.Generator().manual_seed(d.SEED+118011));cells=[]
 for tx in range(6):
  for day in d.DAYS:
   for rx in d.RX:
    ix=order[(L['y'][order]==tx)&(L['day'][order]==day)&(L['rx'][order]==rx)]
    if len(ix)!=70:raise ValueError('Expected70 physical packets per source cell')
    cells.append(ix)
 return torch.stack(cells).T

def derivative(loss,params):
 gs=torch.autograd.grad(loss,params,allow_unused=True,retain_graph=True)
 values=[torch.zeros_like(p).flatten() if g is None else g.detach().flatten() for p,g in zip(params,gs)]
 if any(not torch.isfinite(v).all() for v in values):raise ValueError('Nonfinite actual parameter derivative')
 return values

def gradient_stats(a,b):
 aa=float(a.square().sum());bb=float(b.square().sum());dot=float(a@b)
 return dict(CE_norm=aa**.5,auxiliary_norm=bb**.5,auxiliary_over_CE=(bb/aa)**.5 if aa>0 else None,cosine=dot/(aa*bb)**.5 if aa*bb>0 else None,orthogonal_fraction=(max(0.,1-dot*dot/(aa*bb)))**.5 if aa*bb>0 else None)

def analyze(raw):
 summaries=[]
 for name in d.LEVELS:
  q=raw['triples'];y,rx,day=(raw[k] for k in ('y','rx','day'));gap=raw['relations'][name]['gap'];delta=raw['relations'][name]['delta'];active=gap>0;wrong=delta<0;unique=torch.unique(q[active].flatten());groups=[]
  for tx in range(6):
   for neg in range(6):
    if tx==neg:continue
    keep=(y[q[:,0]]==tx)&(y[q[:,2]]==neg);groups.append(dict(anchorTX=tx,negativeTX=neg,relations=int(keep.sum()),unsatisfied=int((active&keep).sum()),wrong_order=int((wrong&keep).sum()),active_packets=int(torch.unique(q[active&keep].flatten()).numel())))
  rxgroups=[]
  for r in d.RX:
   for s in d.RX:
    if r==s:continue
    keep=(rx[q[:,0]]==r)&(rx[q[:,1]]==s);rxgroups.append(dict(anchorRX=r,referenceRX=s,relations=int(keep.sum()),unsatisfied=int((active&keep).sum()),wrong_order=int((wrong&keep).sum())))
  days=[]
  for dd in d.DAYS:
   keep=day[q[:,0]]==dd;days.append(dict(day=dd,relations=int(keep.sum()),unsatisfied=int((active&keep).sum()),wrong_order=int((wrong&keep).sum()),active_packets=int(torch.unique(q[active&keep].flatten()).numel())))
  summaries.append(dict(level=name,relations=len(q),unsatisfied_relations=int(active.sum()),unsatisfied_rate=float(active.double().mean()),wrong_order_relations=int(wrong.sum()),active_physical_packets=len(unique),active_packet_rate=len(unique)/6300,source_packets=6300,margin=d.MARGIN,mean_hinge=float(gap.clamp_min(0).double().mean()),TX_pairs=groups,RX_pairs=rxgroups,days=days))
 gradients=[]
 for name in ('t1','t2','joint160','SHALLOW_MIDDLE_ORDER','SHALLOW_MIDDLE_SUPCON'):
  for layer in range(3):
   a=raw['CE_gradients'][layer];b=raw['aux_gradients'][name][layer];stats=[gradient_stats(x,y) for x,y in zip(a,b)];gradients.append(dict(objective=name,parameter=d.PARAMETERS[layer],cohort_count=len(stats),nonzero_cohorts=sum(s['auxiliary_norm']>1e-12 for s in stats),mean_auxiliary_norm=sum(s['auxiliary_norm'] for s in stats)/len(stats),mean_CE_norm=sum(s['CE_norm'] for s in stats)/len(stats),mean_ratio=sum(s['auxiliary_over_CE'] or 0 for s in stats)/len(stats),mean_cosine=sum(s['cosine'] for s in stats if s['cosine'] is not None)/sum(s['cosine'] is not None for s in stats) if any(s['cosine'] is not None for s in stats) else None))
 return dict(relation_summary=summaries,gradient_summary=gradients,source_packets=6300,cohorts=70,triplets_per_view=len(raw['triples']),gradient_records=70*5*3,view_shapes=raw['view_shapes'],optimizer_updates=0,head_fit_updates=0,query_read=False,U_truth_read=False)

def measure(L,m):
 ix=cohorts(L);m.eval().requires_grad_(True);named=dict(m.named_parameters());params=[named[n] for n in d.PARAMETERS];head=m.id_backbone.encoder.core.id_backbone.cls_head.weight
 saved={k:v.clone() for k,v in m.state_dict().items()};rng=torch.get_rng_state().clone();start=time.time()
 raw=dict(y=L['y'],rx=L['rx'],day=L['day'],source_physical_ids=L['ids'],cohort_indices=ix,triples=[],relations={n:dict(gap=[],delta=[]) for n in d.LEVELS},CE_gradients=[[] for _ in params],aux_gradients={n:[[] for _ in params] for n in (*d.LEVELS,'SHALLOW_MIDDLE_ORDER','SHALLOW_MIDDLE_SUPCON')},view_shapes={},config=d.config(d.rows()[0]))
 with numerical_context(d.FULL_FP32_POLICY):
  for rank,index in enumerate(ix):
   y,rx,day=(L[k][index] for k in ('y','rx','day'));ids=[L['ids'][i] for i in index.tolist()];q,groups=relation_indices(y,rx,day,ids)
   if len(q)!=1800 or len(groups)!=360:raise ValueError('Fixed source exposure differs')
   with capture(m) as maps:out=m.forward_identity_only(L['x'][index])
   z=views(maps,out);order={};supcon={}
   ce=F.cross_entropy(out['tx_logits'],y,label_smoothing=.01);cg=derivative(ce,params)
   for layer,g in enumerate(cg):raw['CE_gradients'][layer].append(g)
   for name in d.LEVELS:
    order[name],supcon[name],relation=losses(z[name],q,groups);raw['view_shapes'][name]=list(z[name].shape)
    for key in ('gap','delta'):raw['relations'][name][key].append(relation[key])
   targets={n:d.COEFFICIENT*order[n] for n in d.LEVELS};targets['SHALLOW_MIDDLE_ORDER']=d.COEFFICIENT*(order['t1']+order['t2'])/2;targets['SHALLOW_MIDDLE_SUPCON']=d.COEFFICIENT*(supcon['t1']+supcon['t2'])/2
   for name,loss in targets.items():
    if not torch.isfinite(loss):raise ValueError('Nonfinite source auxiliary loss')
    if torch.autograd.grad(loss,head,allow_unused=True,retain_graph=True)[0] is not None:raise ValueError('Auxiliary depends on class W')
    for layer,g in enumerate(derivative(loss,params)):raw['aux_gradients'][name][layer].append(g)
   raw['triples'].append(index[q]);del out,z,order,supcon,targets,ce,cg
   if (rank+1)%10==0:print(json.dumps(dict(source_cohorts=rank+1,source_packets=(rank+1)*90)),flush=True)
 if any(not torch.equal(v,m.state_dict()[k]) for k,v in saved.items()) or any(p.grad is not None for p in m.parameters()) or not torch.equal(rng,torch.get_rng_state()):raise ValueError('Read-only model/gradient/RNG changed')
 raw['triples']=torch.cat(raw['triples'])
 for name in d.LEVELS:
  for key in ('gap','delta'):raw['relations'][name][key]=torch.cat(raw['relations'][name][key])
 raw['CE_gradients']=[torch.stack(g) for g in raw['CE_gradients']]
 for name in raw['aux_gradients']:raw['aux_gradients'][name]=[torch.stack(g) for g in raw['aux_gradients'][name]]
 return raw,dict(seconds=time.time()-start,source_forwards=6300,parameters_buffers_unchanged=True,global_RNG_unchanged=True,accumulated_gradients=False,auxiliary_class_W_gradient=False)

def smoke(m):
 saved={k:v.clone() for k,v in m.state_dict().items()};m.requires_grad_(True)
 with capture(m) as maps:out=m.forward_identity_only(torch.zeros(90,2,256))
 if views(maps,out)['joint160'].shape!=(90,160) or any(not torch.equal(v,m.state_dict()[k]) for k,v in saved.items()):raise ValueError('Real checkpoint no-query smoke failed')
 return dict(status='PASS',zero_IQ_packets=90,head_fit_updates=0,optimizer_updates=0,query_read=False)

def main():
 from experiments.cvs_nocfo_foundation.source import load_source
 from experiments.cvs_nocfo_foundation import design as nd
 torch.set_num_threads(2)
 if d.BASE.exists():raise FileExistsError('Existing immutable run, reconcile')
 d.BASE.mkdir(parents=True);out=d.BASE/d.ROW;out.mkdir()
 try:
  source_proof(d.source_config(),design=pd);L,U,V,contract=load_source(nd.config(nd.rows()[0]));assert U is None;del V
  raw,evidence=measure(L,model());torch.save(raw,out/'source_relations_gradients.pt');result=analyze(raw);recount=analyze(torch.load(out/'source_relations_gradients.pt',map_location='cpu',weights_only=False))
  if result!=recount:raise ValueError('Saved source relations/gradient recount differs')
  result.update(status='VERIFIED',run_id=d.RUN,row_id=d.ROW,evidence=evidence,saved_artifact_recount='VERIFIED',scope='All fixed L physical packets but126000 sampled balanced relations, not all possible triples. Final50 eval derivatives only, not early/mid actual training, no target/Phase2/newTX efficacy or causalRXinvariance.')
  d.write(out/'results.json',result);d.write(out/'resolved_config.json',d.config(d.rows()[0]));done=dict(status='ANALYZED',models=1,source_only=True,source_packets=6300,query_read=False,U_truth_read=False,optimizer_updates=0,head_fit_updates=0,recount='VERIFIED');d.write(out/'completion.json',done);d.write(d.BASE/'completion.json',done);print(json.dumps(done),flush=True)
 except Exception as error:d.write(d.BASE/'failure.json',dict(status='FAILED',error=repr(error),traceback=traceback.format_exc(),outputs_preserved=True));raise
if __name__=='__main__':main()
