"""Ordinary cosine diagnostics on the frozen deployed embedding. Phase1 only.

Source caches and probe checkpoints stay in this run; Phase2 must never load them.
Every fitted parameter uses L_s only. V is diagnostic, never model selection.
"""
import time
from pathlib import Path
import torch
from torch import nn
import torch.nn.functional as F
from . import design as d
from .features import embedding,evidence_mask,GROUPS
from .provenance import own_model,source_proof

class CosineProbe(nn.Module):
 def __init__(self,classes=6,dim=160,seed=0,scale=30.):
  super().__init__()
  with torch.random.fork_rng(devices=[]):
   torch.manual_seed(seed);self.weight=nn.Parameter(torch.randn(classes,dim)*.01)
  self.scale=float(scale)
 def forward(self,z):return self.scale*F.linear(F.normalize(z,dim=1),F.normalize(self.weight,dim=1))

def fit(z,y,classes,seed,settings,role='L_s'):
 if role!='L_s':raise ValueError('Probe fitting is restricted to L_s')
 if settings!=d.PROBE or z.ndim!=2 or z.shape[1]!=160 or len(y)!=len(z) or len(y)==0 or y.min()<0 or y.max()>=classes:raise ValueError('Fixed probe inputs differ')
 # A detached cache and optimizer containing only W enforce a frozen backbone.
 z=z.detach();y=y.detach();head=CosineProbe(classes,160,seed,settings['scale']).to(z.device)
 optimizer=torch.optim.AdamW(head.parameters(),lr=settings['lr'],weight_decay=settings['weight_decay'])
 rng=torch.Generator().manual_seed(seed+1);losses=[];updates=0
 for epoch in range(settings['epochs']):
  order=torch.randperm(len(z),generator=rng).tolist();total=0.
  for start in range(0,len(order),settings['batch_size']):
   ix=order[start:start+settings['batch_size']];optimizer.zero_grad(set_to_none=True)
   loss=F.cross_entropy(head(z[ix]),y[ix]);loss.backward()
   if not torch.isfinite(loss) or not torch.isfinite(head.weight.grad).all():raise FloatingPointError('Source probe nonfinite; no retry')
   optimizer.step();updates+=1;total+=float(loss.detach())*len(ix)
  losses.append(total/len(z))
 head.eval().requires_grad_(False)
 return head,dict(fit_role='L_s',selection='final_only',epochs=settings['epochs'],updates=updates,fit_count=len(z),loss_curve=losses,backbone_updates=0)

def rates(head,z,y,rx):
 with torch.no_grad():pred=head(z).argmax(1);correct=(pred==y)
 groups={str(r):float(correct[rx==r].float().mean()) for r in sorted(rx.unique().tolist())}
 return dict(count=len(y),accuracy=float(correct.float().mean()),receiver_accuracy=groups,worst_receiver=min(groups.values()))

def source_cache(c,model):
 from torch.utils.data import DataLoader
 from experiments.cvs_phase1_stack.runtime import installed
 from experiments.cvs_phase1_stack.fast_execution import configure
 from cvsrffi.xuc_fusion.native import role_ids_from_native
 a=configure(d.make_args(c));expected=d.read(d.SOURCE);values={};device=next(model.parameters()).device
 with installed(c) as n:
  ctx=n._build_ssdg_wisig_data(a,device)
  if ctx['named_test_loaders'] or role_ids_from_native(ctx)!=expected['role_ids']:raise ValueError('Frozen probes require exact source roles and no target loaders')
  for role,key,count in [('L_s','train_loader',6300),('V','val_loader',27000)]:
   original=ctx[key]
   loader=DataLoader(original.dataset,batch_size=256,shuffle=False,drop_last=False,num_workers=0,collate_fn=original.collate_fn)
   zz=[];yy=[];rr=[]
   if len(loader.dataset)!=count:raise ValueError('Full source role count differs')
   with torch.no_grad():
    for x,y,domain,meta in loader:
     z=embedding(model,x.to(device))
     if z.shape!=(len(y),160) or not torch.isfinite(z).all() or y.min()<0 or y.max()>=6:raise ValueError('Source embedding/visible TX differs')
     zz.append(z.cpu());yy.append(y.cpu());rr.append(meta['rx_i'].cpu())
   values[role]=dict(z=torch.cat(zz),y=torch.cat(yy),rx=torch.cat(rr),physical_ids=expected['role_ids'][role])
   if role=='V':
    # Four deterministic source-only feature ablations; the full-feature W is
    # held fixed. No ablation runs at target inference or teacher/U training.
    values[role]['partial_features']={}
    for group in GROUPS:
     parts=[]
     with torch.no_grad(),evidence_mask(model,group,role='V',diagnostic=True):
      for x,y,domain,meta in loader:
       z=embedding(model,x.to(device))
       if z.shape!=(len(y),160) or not torch.isfinite(z).all():raise ValueError('Source partial-feature diagnostic differs')
       parts.append(z.cpu())
     values[role]['partial_features'][group]=torch.cat(parts)

   if values[role]['z'].shape!=(count,160) or set(values[role]['rx'].tolist())!={1,3,4,6,8}:raise ValueError('Source probe receiver/count differs')
 # Sorted physical_ids attest the set, not an embedding-order lookup. No index
 # or sample cache is exported to Phase2.
 return values

def run(row):
 c=d.config(next(r for r in d.rows() if r['row_id']==row));source_proof(c);out=d.BASE/row/'probe'
 if out.exists():raise FileExistsError('Existing source probe: reconcile rather than overwrite')
 out.mkdir(parents=True);torch.set_num_threads(2);start=time.perf_counter()
 from experiments.cvs_equivariant_identity.precision import numerical_context
 with numerical_context(d.FULL_FP32_POLICY):
  model=own_model(c,'cuda:0');cache=source_cache(c,model)
  if any(p.requires_grad for p in model.parameters()) or any(p.grad is not None for p in model.parameters()):raise ValueError('Backbone is not frozen')
  torch.save(cache,out/'phase1_source_features.pt');del model
  cache={r:{k:v.to('cuda:0') if torch.is_tensor(v) else v for k,v in fields.items()} for r,fields in cache.items()}
  L,V=cache['L_s'],cache['V'];seed=c['model_seed']+d.PROBE['seed_offset']
  head,cost=fit(L['z'],L['y'],6,seed,d.PROBE);full=rates(head,V['z'],V['y'],V['rx'])
  partial={group:rates(head,z.to('cuda:0'),V['y'],V['rx']) for group,z in V['partial_features'].items()}
  torch.save(dict(weight=head.weight.cpu(),scale=30.,config=c,settings=d.PROBE,fit_role='L_s',fit_count=6300,backbone_frozen=True,embedding='id_feat_joint',readout='REFIT_COSINE',selection='final_only',cost=cost),out/'refit_cosine.pt')
  held=[];costs=[dict(probe='full_TX',**cost)]
  for rx in (1,3,4,6,8):
   lm=L['rx']!=rx;vm=V['rx']==rx
   h,cc=fit(L['z'][lm],L['y'][lm],6,seed+rx,d.PROBE)
   held.append(dict(held_receiver=rx,**rates(h,V['z'][vm],V['y'][vm],V['rx'][vm])))
   costs.append(dict(probe='leave_RX_'+str(rx),**cc))
  rxs=(1,3,4,6,8);ly=torch.empty_like(L['rx']);vy=torch.empty_like(V['rx'])
  for i,rx in enumerate(rxs):ly[L['rx']==rx]=i;vy[V['rx']==rx]=i
  h,cc=fit(L['z'],ly,5,seed+809,d.PROBE);rxprobe=rates(h,V['z'],vy,V['rx']);costs.append(dict(probe='RX_information',**cc))
 result=dict(status='SOURCE_PROBES_FROZEN',config=c,settings=d.PROBE,target_access=False,truth_read=False,fit_roles=['L_s'],diagnostic_roles=['V'],head_selection='final_only',backbone_updates=0,full_TX=full,partial_source_TX=partial,partial_diagnostic='fixed full-feature cosineW on sourceV with one group removed; no target masks or extra fitting',leave_RX=held,leave_RX_mean=sum(v['accuracy'] for v in held)/5,leave_RX_worst=min(v['accuracy'] for v in held),RX_information=rxprobe,costs=costs,head_epochs_total=70,elapsed_seconds=time.perf_counter()-start,source_cache_phase2_allowed=False)
 d.write(out/'summary.json',result);print('SOURCE_PROBES_FROZEN '+row,flush=True)
 return result

def proof(c):
 source_proof(c);out=d.BASE/c['row_id']/'probe';v=d.read(out/'summary.json')
 if v['status']!='SOURCE_PROBES_FROZEN' or v['config']!=c or v['settings']!=d.PROBE or v['target_access'] or v['truth_read'] or v['backbone_updates'] or v['fit_roles']!=['L_s'] or v['diagnostic_roles']!=['V'] or v['head_selection']!='final_only' or v['source_cache_phase2_allowed']:raise ValueError('Frozen source probe provenance differs')
 ck=torch.load(out/'refit_cosine.pt',map_location='cpu',weights_only=False)
 if ck['config']!=c or ck['settings']!=d.PROBE or ck['fit_role']!='L_s' or ck['fit_count']!=6300 or not ck['backbone_frozen'] or ck['embedding']!='id_feat_joint' or ck['readout']!='REFIT_COSINE' or ck['selection']!='final_only' or ck['scale']!=30. or ck['weight'].shape!=(6,160) or not torch.isfinite(ck['weight']).all():raise ValueError('Ordinary probe checkpoint differs')
 return ck,v
