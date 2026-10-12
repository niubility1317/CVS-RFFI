import argparse,copy,json,tempfile,torch
from pathlib import Path
from unittest.mock import patch
from . import design as d
from .response import Bank
from .objective import objective,eval_forward
from .model import groups,expected_parent,build
from .provenance import initialization,checkpoint_provenance
from experiments.cvs_nocfo_engineering.teacher import teacher_config,expected_proof
from experiments.cvs_nocfo_foundation.model import build as native
from experiments.cvs_finite_action import design as fd
from experiments.cvs_finite_action.checks import synthetic_L
from experiments.cvs_finite_action.predictors import Action,normfit
from experiments.cvs_finite_action.operators import quality,apply
def fixture(model,root,c,device):
 gen=torch.Generator().manual_seed(128);x=torch.randn(16,2,256,generator=gen);L=synthetic_L();L['x']=x.repeat(394,1,1)[:6300];U=dict(role='U_s',ids=[f'U-{i:06}' for i in range(56700)],x=x.repeat(3544,1,1)[:56700])
 _,h=eval_forward(model,x.to(device),features=True,gradient=False);q=quality(x)
 targets={}
 for kind,a in c['response']['known_parameters'].items():
  _,acted=eval_forward(model,apply(x.to(device),kind,a,1),features=True,gradient=False);targets[kind]={level:(acted[level]-h[level]).cpu() for level in h}
 fc=fd.config(next(r for r in fd.rows() if r['model_seed']==c['model_seed']));root.mkdir()
 from experiments.cvs_finite_action.splits import make
 qL=q.repeat(394,1)[:6300];split,_=make(L,qL,fc);fi,ev=split['PID80_20'];observed={}
 for key,data in [('L',L),('U',U)]:
  n=len(data['ids']);observed[key]=dict(role=data['role'],source_only=True,U_truth=False,ids=data['ids'],q=q.repeat((n+15)//16,1)[:n],targets={})
  for kind in targets:
   points=list(range(len(fc['points'][kind]))) if key=='L' else [1]
   observed[key]['targets'][kind]=dict(points=points,mean={level:torch.stack([t.repeat((n+15)//16,1)[:n] for _ in points],1) for level,t in targets[kind].items()})
 torch.save(observed,root/'source_observations.pt');parent=expected_parent(c)
 fd.write(root/'completion.json',dict(config=fc,status='SOURCE_DIAGNOSTIC_COMPLETE',checkpoint_provenance=parent,identity_updates=0,head_updates=0,target_access=False,budget=dict(reference_parameters_buffers_modes_gradients_unchanged=True),auxiliary_fits=117,auxiliary_optimizer_steps=117180))
 fd.write(root/'resolved_config.json',dict(config=fc,commit=d.FINITE_EXECUTION_COMMIT,checkpoint_provenance=parent));fd.write(root.parent/'completion.json',dict(status='SOURCE_DIAGNOSTIC_COMPLETE',rows=[r['row_id'] for r in fd.rows()],identity_updates=0,target_access=False))
 for kind,level,method in d.TASKS:
  state={'h':normfit(h[level].cpu()),'q':normfit(q),'a':normfit(torch.tensor([[-1.,0.],[1.,0.]]*8)),'y':normfit(targets[kind][level])};dim=len(state['h']['mean']);net=Action(dim+6,dim,128,False)
  torch.save(dict(config=fc,split='PID80_20',kind=kind,level=level,noise=False,fit_IDs=[L['ids'][i] for i in fi.tolist()],eval_IDs=[L['ids'][i] for i in ev.tolist()],state=state,net=net.state_dict(),bilinear=torch.randn(dim*3+12,dim,generator=gen)*.01),root/f"PID80_20-{kind}-{level.replace('.','_')}.pt")
 return L,U
def proof(c):
 rc=c['response']
 if not rc['enabled']:return None
 return dict(status='VERIFIED',run_id=fd.RUN,execution_commit=d.FINITE_EXECUTION_COMMIT,seed=c['model_seed'],split='PID80_20',source_only=True,identity_updates=0,target_access=False,reference_parent=expected_parent(c),tasks=rc['tasks'],source_root=rc['source_root'],state_paths=[str(Path(rc['source_root'])/f"PID80_20-{k}-{l.replace('.','_')}.pt") for k,l,m in rc['tasks']],observations_path=str(Path(rc['source_root'])/'source_observations.pt'),auxiliary_trainable_parameters=0)
def run(device):
 torch.set_num_threads(2);results=[];seed=d.SEEDS[0];base=native(teacher_config(seed),device);rc=d.config(next(r for r in d.rows() if r['model_seed']==seed and r['arm']=='RESP_LU'))
 with tempfile.TemporaryDirectory() as tmp:
  root=Path(tmp)/'finite';L,U=fixture(base,root,rc,device)
  for row in [r for r in d.rows() if r['model_seed']==seed]:
   c=d.config(row);c['response']['source_root']=str(root);model=native(teacher_config(seed),device);model.train();pg,_=groups(model,c);opt=torch.optim.AdamW(pg,weight_decay=.0001);ids=torch.arange(16)
   peers={str(s):expected_proof(s) for s in c['teacher']['seeds']};peer=dict(ids=U['ids'],q=torch.full((56700,6),1/6),peer_JS=torch.zeros(56700),provenance=peers) if c['unlabeled'] else None
   bank=Bank(c,L,U,device) if c['response']['enabled'] else None;before={k:{n:t.clone() for n,t in reader.state_dict().items()} for k,reader in bank.readers.items()} if bank else None
   loss,parts,w,details=objective(c,model,peer,bank,L,ids,U if c['unlabeled'] else None,ids if c['unlabeled'] else None,222,device)
   auxnorm=None
   if bank:
    auxgrads=torch.autograd.grad(parts['extra'],[p for g in pg for p in g['params']],allow_unused=True,retain_graph=True);auxnorm=float(sum(g.square().sum() for g in auxgrads if g is not None).sqrt());assert auxnorm>1e-8
   loss.backward();norm=torch.nn.utils.clip_grad_norm_([p for g in pg for p in g['params']],10.);assert torch.isfinite(norm);opt.step()
   if bank:
    assert all(p.grad is None for reader in bank.readers.values() for p in reader.parameters());assert all(torch.equal(t,before[key][name]) for key,reader in bank.readers.items() for name,t in reader.state_dict().items())
   if row['arm']=='CONT_CE':assert details['U_packets']==details['extra_forward_calls']==details['response_readout_calls']==0 and torch.equal(loss,parts['ce'])
   if row['arm']=='ACTION_LU':assert details['actual_IQ_operation'] and details['extra_L_packets']==details['extra_U_packets']==16
   modes=[m.training for m in model.modules()];rng=torch.get_rng_state().clone();buffers={n:b.clone() for n,b in model.named_buffers()}
   eval_forward(model,L['x'][:16].to(device),features=True,gradient=True);assert modes==[m.training for m in model.modules()] and torch.equal(rng,torch.get_rng_state()) and all(torch.equal(b,buffers[n]) for n,b in model.named_buffers())
   if c['unlabeled']:
    try:objective(c,model,peer,bank,L,ids,dict(U,y=torch.zeros(56700,dtype=torch.long)),ids,222,device)
    except ValueError:pass
    else:raise AssertionError('U truth accepted')
   results.append(dict(arm=row['arm'],finite_one_identity_update=True,response_input_grad_norm=auxnorm,reader_frozen=bool(bank),eval_modes_buffers_rng_preserved=True,U_truth_rejected=c['unlabeled']))
  strict=[]
  for row in d.rows():
   c=d.config(row);init=initialization(c,expected_parent(c),{str(s):expected_proof(s) for s in c['teacher']['seeds']},proof(c));payload=dict(init);payload.update(config=c,model_seed=c['model_seed'],epoch=20,total_epochs=40,parent_training_epochs=20,incremental_epochs=20,optimizer_steps=4440,lifetime_optimizer_steps=8880,run_id=d.RUN,candidate_id=c['row_id'],parent_checkpoint=c['parent_checkpoint'],checkpoint_selection='final_only',target_access=False,scratch_only=False,reference_E20_loaded=True,model=base.state_dict(),baseline_args=base._stage2_baseline_args)
   checkpoint_provenance(c,payload)
   for field,bad in [('total_epochs',20),('parent_checkpoint','unknown'),('target_access',True),('model_seed',1)]:
    try:checkpoint_provenance(c,dict(payload,**{field:bad}))
    except ValueError:pass
    else:raise AssertionError('Invalid own checkpoint accepted')
   if row['arm']=='CONT_CE' and row['model_seed']==seed:
    path=Path(tmp)/'own.pth';torch.save(payload,path)
    with patch('experiments.cvs_e20_response.model.verify',side_effect=AssertionError('Own prediction reloaded parent')):own=build(c,device,path)
    assert all(torch.equal(t,own.state_dict()[n]) for n,t in base.state_dict().items())
   strict.append(row['row_id'])
 return dict(status='PASS',device=device,synthetic_only=True,rows=results,strict_own_checkpoint_rows=len(strict),known_real_IQ_operations=True,identity_updates=7,target_access=False)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--device',default='cpu');p.add_argument('--output',required=True,type=Path);a=p.parse_args();v=run(a.device);d.write(a.output,v);print(json.dumps(v))
