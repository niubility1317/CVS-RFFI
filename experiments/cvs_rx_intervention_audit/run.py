import os,sys,time
from pathlib import Path
import torch
from torch.nn import functional as F
from . import design as d
from .actions import fit,coefficients,apply,basis
from experiments.cvs_rx_spectral_audit.spectral import observations
from experiments.cvs_stage2_teacher_factorial.model import build,groups
from experiments.cvs_source_anchor_audit.run import extract,paired
from experiments.cvs_equivariant_identity.precision import numerical_context,actual_flags

def load(c):
 from experiments.cvs_rx_spectral_audit.run import load_l
 from experiments.cvs_stage2_decouple.common import verify_parent
 L,provenance=load_l(dict(preparation=c['parent_build_config']['preparation']))
 provenance['parent']=verify_parent(c['parent_build_config']['parent_contract'])
 return L,provenance

def gradients(model,x,y,parameter_groups):
 loss=F.cross_entropy(model(x),y);parameters=[p for g in parameter_groups for p in g['params']]
 grads=torch.autograd.grad(loss,parameters,allow_unused=True);result={};offset=0
 for group in parameter_groups:
  count=len(group['params']);values=grads[offset:offset+count];offset+=count
  result[group['name']]=torch.cat([(torch.zeros_like(p) if g is None else g).detach().reshape(-1) for p,g in zip(group['params'],values)])
 return result

def run(row_id,device='cuda:0'):
 c=d.config(next(r for r in d.rows() if r['row_id']==row_id));d.validate(c);out=Path(c['output']);out.mkdir(parents=True,exist_ok=False);started=time.perf_counter();torch.set_num_threads(2)
 L,provenance=load(c);state=fit(L);d.write(out/'provenance.json',provenance)
 d.write(out/'resolved_config.json',dict(config=c,pid=os.getpid(),cwd=os.getcwd(),python=sys.executable,commit=(d.ROOT/'release_commit.txt').read_text().strip(),CUDA_VISIBLE_DEVICES=os.environ.get('CUDA_VISIBLE_DEVICES'),target_loaders_constructed=False))
 d.write(out/'action_fit.json',dict(labels=state['labels'],effects=state['effects'].tolist(),fit='crossfit otherTX AND otherday',U_used=False,V_used=False,target_access=False))
 with numerical_context(d.parent.FULL_FP32_POLICY):
  model=build(c['parent_build_config'],device).eval();initial={n:t.detach().clone() for n,t in model.state_dict().items()};torch.cuda.reset_peak_memory_stats()
  pg,_=groups(model,c['parent_build_config']);names={id(p):n for n,p in model.named_parameters()}
  pg=[dict(name=g['name'],params=[p for p in g['params'] if not names[id(p)].endswith(('low_hz_','band_hz_'))]) for g in pg]
  clean_f,clean_q=extract(model,L,device);y=L['y'].to(device);clean_log=clean_q.clamp_min(1e-12).log();base_obs=observations(L['x'],'rectangular')['centered_log_PSD32']
  audit_ids=[]
  for tx in state['TX']:
   for day in state['day']:
    for rx in state['RX']:audit_ids.append(int(torch.where((L['y']==tx)&(L['day']==day)&(L['rx']==rx))[0][0]))
  idx=torch.tensor(audit_ids);gx=L['x'][idx].to(device);gy=L['y'][idx].to(device);g0=gradients(model,gx,gy,pg);reports=[]
  for variant in c['intervention']['variants']:
   for scale in c['intervention']['scales']:
    for shift in c['intervention']['RX_shifts']:
     delta=coefficients(state,L['y'],L['rx'],L['day'],shift,variant['kind']);features=[];probabilities=[];observed=[];diagnostics=[];recovered=[]
     with torch.no_grad():
      for start in range(0,len(L['x']),256):
       x=L['x'][start:start+256].to(device);co=delta[start:start+256].to(device);z,info=apply(x,co,scale,variant['taps'])
       result=model.forward_identity_only(z);features.append(F.normalize(result['id_feat_joint'],dim=1));probabilities.append(result['tx_logits'].softmax(1))
       observed.append(observations(z,'rectangular')['centered_log_PSD32'].cpu());diagnostics.append({k:v.detach().cpu() for k,v in info.items()})
       inverse,_=apply(z,-co,scale,variant['taps']);half=variant['taps']-1
       recovered.append(((inverse[:,:,half:-half]-x[:,:,half:-half]).square().sum((1,2))/x[:,:,half:-half].square().sum((1,2)).clamp_min(1e-12)).cpu())
     f=torch.cat(features);q=torch.cat(probabilities);logq=q.clamp_min(1e-12).log();diag={k:torch.cat([r[k] for r in diagnostics]) for k in diagnostics[0]}
     intended=diag['target_log_power'].reshape(-1,32,8).mean(-1).double();realized=torch.cat(observed)-base_obs;den=float(intended.square().sum())
     transformed,_=apply(gx,delta[idx].to(device),scale,variant['taps']);g1=gradients(model,transformed,gy,pg)
     gradient={name:dict(clean_norm=float(g0[name].norm()),intervention_norm=float(g.norm()),cosine=float(F.cosine_similarity(g[None],g0[name][None])),relative_change=float((g-g0[name]).norm()/g0[name].norm().clamp_min(1e-12))) for name,g in g1.items()}
     p=paired(q,clean_q,y);by_TX={str(tx):paired(q[y==tx],clean_q[y==tx],y[y==tx]) for tx in y.unique().tolist()}
     record=dict(variant=variant['name'],scale=scale,RX_shift=shift,identity=p,TX=by_TX,NLL=float(-logq[torch.arange(len(y),device=device),y].mean()),clean_NLL=float(-clean_log[torch.arange(len(y),device=device),y].mean()),feature_cosine_mean=float((f*clean_f).sum(1).mean()),feature_cosine_p05=float((f*clean_f).sum(1).quantile(.05)),physical=dict(observable_shift_relative_MSE=float((realized-intended).square().sum())/den if den else None,observable_shift_cosine=float(F.cosine_similarity(realized.reshape(1,-1),intended.reshape(1,-1))),FIR_response_relative_MSE=float((diag['effective_log_power']-diag['target_log_power']).square().sum()/diag['target_log_power'].square().sum().clamp_min(1e-12)),clipped_bin_fraction=float(diag['clipped_fraction'].mean()),boundary_wrap_comparison_MSE=float(diag['boundary_residual_fraction'].mean()),interior_wrap_comparison_MSE=float(diag['interior_residual_fraction'].mean()),roundtrip_interior_relative_MSE=float(torch.cat(recovered).mean())),gradient=gradient,gradient_count=len(idx))
     reports.append(record);d.write(out/'intervention_results.json',dict(complete=False,rows=reports));print(variant['name'],scale,shift,'acc',p['accuracy'],'breaks',p['break_count'],'physical_MSE',record['physical']['observable_shift_relative_MSE'],flush=True)
  assert all(torch.equal(initial[n],t) for n,t in model.state_dict().items())
  d.write(out/'intervention_results.json',dict(complete=True,rows=reports,clean_accuracy=float((clean_q.argmax(1)==y).float().mean()),clean_count=len(y),readout_scope='source L training samples, not held-out classification performance',fit_scope='action banks crossfit TX AND day',U_used=False,V_used=False,target_access=False))
  done=dict(status='SOURCE_DIAGNOSTIC_COMPLETE',row_id=row_id,identity_updates=0,target_access=False,L_count=len(y),views=len(reports),source_model_unchanged=True,elapsed_seconds=time.perf_counter()-started,peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),backend=actual_flags())
  d.write(out/'completion.json',done);print(done,flush=True)
