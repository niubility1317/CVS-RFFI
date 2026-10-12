import os,sys,time
from pathlib import Path
import torch
from . import design as d
from .statistics import fit
from experiments.cvs_scratch30_baseline.predict import own_model
from experiments.cvs_nocfo_foundation.source import load_source
from experiments.cvs_equivariant_identity.precision import numerical_context,actual_flags
def run(seed,device='cuda:0'):
 if seed not in d.SEEDS:raise ValueError('Unregistered parentseed')
 root=d.BASE/('s'+str(seed));root.mkdir(exist_ok=False);c=d.parent(seed);torch.set_num_threads(2);started=time.perf_counter();L,U,V,contract=load_source(c['source_loader_config'])
 if U is not None or len(L['ids'])!=6300:raise ValueError('Projection Lonly data contract')
 with numerical_context(d.FULL_FP32_POLICY):
  model=own_model(c,device);before={k:v.clone() for k,v in model.state_dict().items()};hh=[]
  with torch.no_grad():
   for begin in range(0,6300,256):hh.append(torch.nn.functional.normalize(model.forward_identity_only(L['x'][begin:begin+256].to(device))['id_feat_joint'],dim=1,eps=1e-4).cpu())
  if any(not torch.equal(v,model.state_dict()[k]) for k,v in before.items()):raise ValueError('Sourcefit changedfrozenparent')
  states,report=fit(torch.cat(hh),L['y'],L['rx'],model.head.weight.detach().cpu())
  for name,state in states.items():
   row=next(r for r in d.rows() if r['arm']==name and r['model_seed']==seed);cfg=d.config(row);out=Path(cfg['output']);out.mkdir(exist_ok=False);torch.save(dict(config=cfg,state=state,fit_ids=L['ids'],parent_config=c,epoch=30,new_gradient_updates=0,ancestors=[c['run_id']],target_access=False),out/'model.pth')
  d.write(root/'source_contract.json',contract);d.write(root/'source_fit_diagnostics.json',report);d.write(root/'resolved_config.json',dict(pid=os.getpid(),cwd=os.getcwd(),python=sys.executable,commit=(d.ROOT/'release_commit.txt').read_text().strip(),parent_config=c,fit_L=6300,U_opened=False,V_forward=False,target_access=False,parameter_updates=0,identity_input_policy=model._identity_input_policy,backend_flags=actual_flags(),elapsed_seconds=time.perf_counter()-started))
  d.write(root/'source_frozen.json',dict(status='SOURCE_FROZEN',seed=seed,methods=list(d.METHODS),fit_L=6300,fit_ids=L['ids'],parent_config=c,gradient_updates=0,parent_epochs=30,target_access=False))
 print('SOURCE_FROZEN '+str(seed),flush=True)
