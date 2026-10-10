"""Pre-training source mechanism audit on fixed, provenance-checked identity models."""
import argparse,json,os,time
from pathlib import Path
from types import SimpleNamespace
import torch
from torch.nn import functional as F
from . import design as d
from .actions import StateAction,NeuralPairEstimator,estimate_pair,action_loss,conditional_edges,chain_metrics,proposal_select,verify_proposal
from .physics import factorial_batch,cfo_additive_diagnostic,quality_proxies,contract
from experiments.cvs_multi_disentangle.model import intermediate,classify_intermediate,_encoder
from experiments.cvs_multi_action_audit.runner import role_split
from experiments.cvs_multi_state_action import design as old
from experiments.cvs_phase1_stack.runtime import installed
from experiments.cvs_legacy_no_sat.evaluate import source_provenance,checkpoint_provenance
from experiments.cvs_equivariant_identity.precision import numerical_context
from experiments.cvs_multi_action_audit.actions import fixed_identity
from experiments.cvs_multi_state_action.actions import validate_roles

def reference_config(seed):return old.config(next(r for r in old.rows() if r['arm']=='native' and r['model_seed']==seed))
def source_packets(native,c,device):
 from cvsrffi.xuc_fusion.native import role_ids_from_native
 from cvsrffi.game_tracking.data import opaque_id
 ctx=native._build_ssdg_wisig_data(old.make_args(c,str(device)),device)
 expected=d.read(d.SOURCE)
 if ctx['named_test_loaders'] or role_ids_from_native(ctx)!=expected['role_ids']:raise ValueError('Source role mismatch')
 ds=ctx['train_loader'].dataset
 records=[dict(id=opaque_id(v),index=i,role='L_s',y=int(v.tx_i),rx=int(v.rx_i),day=int(v.day_i)) for i,v in enumerate(ds.index)]
 fit,audit=role_split(records,expected['role_ids']['L_s'],d.DIAGNOSTIC['fit_per_tx_rx'],d.DIAGNOSTIC['audit_per_tx_rx'],d.DIAGNOSTIC['seed'])
 def load(rows):
  return dict(x=torch.stack([ds[r['index']][0] for r in rows]).to(device),
   y=torch.tensor([r['y'] for r in rows],device=device),rx=torch.tensor([r['rx'] for r in rows],device=device),
   day=torch.tensor([r['day'] for r in rows],device=device),ids=[r['id'] for r in rows],condition=['clean']*len(rows))
 return load(fit),load(audit),dict(fit=fit,audit=audit,source_contract=d.SOURCE,source_roles='EXACT_MATCH',
   U_labels_read=False,V_read=False,target_read=False,scope='auxiliary holdout; reference identity has seen source L')

def slice_data(data,ix):
 indexes=ix.tolist()
 return {k:v[ix] if torch.is_tensor(v) else [v[i] for i in indexes] for k,v in data.items()}

def action_audit(identity,fit,audit,seed,heldout=None,edges=True,steps=120):
 """Scoped source audit preserves identity state, mode and caller RNG streams."""
 validate_roles(fit,audit)
 before={k:v.detach().clone() for k,v in identity.state_dict().items()}
 with fixed_identity(identity):
  result=_action_audit(identity,fit,audit,seed,heldout,edges,steps)
 if any(not torch.equal(v,identity.state_dict()[k]) for k,v in before.items()):
  raise ValueError('Source action audit mutated frozen identity')
 return result

def _action_audit(identity,fit,audit,seed,heldout=None,edges=True,steps=120):
 device=fit['x'].device;gen=torch.Generator().manual_seed(seed)
 if heldout is not None:fit=slice_data(fit,torch.where(fit['y']!=heldout)[0])
 if heldout is not None and bool((fit['y']==heldout).any()):raise ValueError('TX holdout failed')
 if len(fit['x'])==0 or len(audit['x'])==0 or steps<1:raise ValueError('Empty source audit role/budget')
 # CPU module initialization is isolated; .to(device) consumes no random draws.
 with torch.random.fork_rng(devices=[]):
  torch.random.default_generator.manual_seed(seed)
  actions=torch.nn.ModuleDict({k:StateAction(k,hidden=d.RECIPE['hidden']) for k in ('linear','temporal')}).to(device)
  estimators=torch.nn.ModuleDict({k:NeuralPairEstimator(k) for k in actions}).to(device)
 opt=torch.optim.AdamW(list(actions.parameters())+list(estimators.parameters()),lr=2e-4)
 ref=_encoder(identity).response;logs=[]
 for step in range(steps):
  ix=torch.randperm(len(fit['x']),generator=gen)[:32].to(device);x=fit['x'][ix];y=fit['y'][ix]
  b=factorial_batch(x,gen,step=step)
  with torch.no_grad():h={k:intermediate(identity,b[k]) for k in ('x00','x10','x01','x11')}
  losses=[];row=dict(step=step+1,order=b['order'],conditional_edges=edges,fit_packets=len(x))
  for edge in conditional_edges(b,h,step=step,enabled=edges):
   k=edge['kind'];pred=actions[k](edge['h'],edge['x'],edge['p'],ref,phase_noise=edge['phase_noise'],
      cached_reference=edge['cached_reference'],cached_endpoint_reference=edge['cached_endpoint_reference'])
   fitloss=action_loss(identity,edge['h'],edge['target_delta'],pred,y)
   estimated=estimators[k](edge['x'],edge['endpoint_x'])
   ploss=((estimated-edge['p'])/estimators[k].scales).square().mean()
   losses.extend([fitloss['loss'],ploss]);row[k+'_fit']=float(fitloss['loss'].detach());row[k+'_parameter_loss']=float(ploss.detach())
  loss=torch.stack(losses).mean();opt.zero_grad(set_to_none=True);loss.backward()
  if not torch.isfinite(loss):raise FloatingPointError('Audit nonfinite')
  torch.nn.utils.clip_grad_norm_(list(actions.parameters())+list(estimators.parameters()),1.);opt.step();logs.append(row)
  if step==0 or (step+1)%25==0:
   print('ACTION_FIT '+json.dumps(dict(row,heldout_tx=heldout)),flush=True)
 observed=[];proposals=[];noise_rows=[]
 evaluation_gen=torch.Generator().manual_seed(seed+10000)
 with torch.no_grad():
  for start in range(0,len(audit['x']),32):
   x=audit['x'][start:start+32];y=audit['y'][start:start+32]
   b=factorial_batch(x,evaluation_gen,step=start//32)
   h={k:intermediate(identity,b[k]) for k in ('x00','x10','x01','x11')}
   chained=chain_metrics(actions,b,h,ref)
   for kind in actions:
    endpoint=b['x10' if kind=='linear' else 'x01'];true=b[kind+'_parameters']
    analytic,info=estimate_pair(x,endpoint,kind,b['phase_noise'])
    learned=estimators[kind](x,endpoint)
    for name,p in [('true',true),('analytic',analytic),('learned',learned)]:
     # Estimated parameters need their own exact frontend endpoint, never truth endpoint leakage.
     pred=actions[kind](h['x00'],x,p,ref,phase_noise=b['phase_noise'])
     target=h['x10' if kind=='linear' else 'x01']-h['x00']
     for tx in y.unique().tolist():
      mask=y==tx;parts=action_loss(identity,h['x00'][mask],target[mask],pred[mask],y[mask])
      observed.append(dict(kind=kind,parameter_source=name,tx=tx,count=int(mask.sum()),
       heldout_tx=heldout,conditional_edges=edges,parameter_normalized_mse=float(((p[mask]-true[mask])/estimators[kind].scales).square().mean()),
       **{k:float(v) for k,v in parts.items()},**chained))
    proposed=proposal_select(actions[kind],identity,h['x00'],x,y,evaluation_gen,reference_fn=ref)
    actual=identity(proposed['x']);random=identity(endpoint)
    proposals.append(dict(kind=kind,**verify_proposal(proposed,actual,y),
     random_real_ce=float(F.cross_entropy(random,y)),selected_error_rate=float((actual.argmax(1)!=y).float().mean()),
     random_error_rate=float((random.argmax(1)!=y).float().mean())))
   # The same L/T signal and same unit noise realization at all SNRs.
   noise=torch.randn(b['x11'].shape,generator=evaluation_gen).to(x)
   noise=noise/noise.square().mean((1,2),keepdim=True).sqrt().clamp_min(1e-10)
   base_logits=identity(b['x11']);base_ce=F.cross_entropy(base_logits,y)
   base_error=(base_logits.argmax(1)!=y).float().mean()
   for snr in d.DIAGNOSTIC['noise_snr_db']:
    delta_noise=noise*b['x11'].square().mean((1,2),keepdim=True).sqrt()*10**(-snr/20)
    xn=b['x11']+delta_noise
    logit=identity(xn);noise_rows.append(dict(snr_db=snr,count=len(y),ce=float(F.cross_entropy(logit,y)),
     error_rate=float((logit.argmax(1)!=y).float().mean()),fixed_LT=True,
     paired_base_ce=float(base_ce),paired_base_error_rate=float(base_error),
     ce_increase=float(F.cross_entropy(logit,y)-base_ce),
     measured_added_noise_snr_db=float(10*torch.log10(b['x11'].square().mean()/delta_noise.square().mean())),
     quality_mean=quality_proxies(xn).mean(0).tolist(),same_noise_realization=True,
     interpretation='Added digital noise stress, not measured receiver SNR'))
 return dict(fit_logs=logs,action_records=observed,proposal_records=proposals,noise_records=noise_rows,
   heldout_tx=heldout,conditional_edges=edges,steps=steps,parameter_note='Analytic T conditioned on known realized noise basis; not blind physical recovery',
   fit_unique_physical_ids=len(set(fit['ids'])),audit_unique_physical_ids=len(set(audit['ids'])),
   fit_tx=sorted(fit['y'].unique().tolist()),audit_tx=sorted(audit['y'].unique().tolist()),
   rng='private fit/evaluation generators; isolated initialization',identity_updated=False,target_access=False)

def run(seed,smoke=False):
 torch.set_num_threads(2);device=torch.device('cpu' if smoke else 'cuda:0');c=reference_config(seed)
 source=source_provenance(old,c);out=d.BASE/('diagnostic-s'+str(seed));started=time.perf_counter()
 with numerical_context(d.FULL_FP32_POLICY),installed(c) as native:
  ck=torch.load(source/'final_ssdg.pth',map_location='cpu',weights_only=False);checkpoint_provenance(old,c,ck)
  model=native.build_baseline_model(SimpleNamespace(**ck['baseline_args']),device)
  model.load_state_dict(ck['model'],strict=True);model.eval();del ck
  with torch.no_grad():z=model(torch.zeros(2,2,256,device=device))
  if z.shape!=(2,6) or not torch.isfinite(z).all():raise ValueError('Checkpoint source smoke')
  if smoke:return dict(status='VERIFIED',seed=seed,checkpoint=str(source/'final_ssdg.pth'),target_read=False)
  out.mkdir(parents=True,exist_ok=False)
  d.write(out/'provenance.json',dict(status='VERIFIED',checkpoint=str(source/'final_ssdg.pth'),
   parent_config=c,selection='fixed three native E200 rows; no target metric consumption',identity_initializes_new_training=False,
   source_contract='EXACT_MATCH',target_access=False,pid=os.getpid(),gpu=os.environ.get('CUDA_VISIBLE_DEVICES')))
  fit,audit,roles=source_packets(native,c,device);d.write(out/'physical_roles.json',roles)
  identity=model.id_backbone;before={k:v.clone() for k,v in identity.state_dict().items()}
  for p in identity.parameters():p.requires_grad_(False)
  d.write(out/'physical_diagnostics.json',dict(contract=contract(),cfo=cfo_additive_diagnostic(fit['x'],fit['y'],fit['rx'],fit['day']),
   quality_std=quality_proxies(fit['x']).std(0).tolist(),quality_names=['repeat_coherence','repeat_fit_residual','phase_dispersion']))
  jobs=[(None,False),(None,True)]+[(tx,True) for tx in d.DIAGNOSTIC['heldout_tx']]
  for tx,edges in jobs:
   name=('all' if tx is None else 'heldout_tx'+str(tx))+('_edges' if edges else '_main')
   result=action_audit(identity,fit,audit,d.DIAGNOSTIC['seed'],tx,edges,d.DIAGNOSTIC['steps'])
   d.write(out/(name+'.json'),result);print('SOURCE_DIAGNOSTIC '+name+' COMPLETE',flush=True)
  from .receiver import receiver_audit
  r=receiver_audit(identity,fit,audit,torch.Generator().manual_seed(d.DIAGNOSTIC['seed']+1),steps=d.DIAGNOSTIC['steps'])
  torch.save(r.pop('state_dict'),out/'receiver_state.pth');d.write(out/'receiver.json',r)
  if any(not torch.equal(v,identity.state_dict()[k]) for k,v in before.items()):raise ValueError('Frozen identity changed')
  d.write(out/'completion.json',dict(status='SOURCE_DIAGNOSTIC_COMPLETE',seed=seed,jobs=len(jobs),identity_unchanged=True,
   target_access=False,elapsed_seconds=time.perf_counter()-started,peak_cuda_bytes=torch.cuda.max_memory_allocated()))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--seed',type=int,required=True);p.add_argument('--smoke',action='store_true');a=p.parse_args();run(a.seed,a.smoke)
