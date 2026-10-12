import json,os,sys,time
from pathlib import Path
import torch
from torch.nn import functional as F
from . import design as d
from .metric import fit,logits
from experiments.cvs_finite_margin.run import subset
from experiments.cvs_nocfo_foundation.source import load_source
from experiments.cvs_nocfo_foundation.model import build
from experiments.cvs_wave_geometry.run import score
from experiments.cvs_finite_action.operators import apply
from experiments.cvs_equivariant_identity.precision import numerical_context

def parent(c,device):
 pc=c['parent_config'];root=Path(pc['output']);done=d.read(root/'completion.json');resolved=d.read(root/'resolved_config.json');frozen=d.read(root/'source_frozen.json')
 if done['config']!=pc or done['status']!='SOURCE_DIAGNOSTIC_COMPLETE' or done['identity_updates']!=4440 or done['checkpoint_sources'] or done['target_access'] or done['U_opened'] or resolved['commit']!=c['parent_execution_commit'] or resolved['checkpoint_loaded'] or resolved['ancestors'] or c['held_rx'] in resolved['training_rx'] or frozen['config']!=pc:raise ValueError('CHECKPOINT_PROVENANCE_UNVERIFIED: source holdout ancestor')
 ck=torch.load(c['parent_checkpoint'],map_location='cpu',weights_only=False)
 if ck['config']!=pc or ck['epoch']!=20 or ck['updates']!=4440 or ck['ancestors'] or ck['target_access'] or ck['training_rx']!=resolved['training_rx']:raise ValueError('Actual fixed source holdout checkpoint differs')
 model=build(pc['source_loader_config'],device);model.load_state_dict(ck['model'],strict=True);model.eval().requires_grad_(False)
 from experiments.cvs_phase1_stack.recover import verify_contract
 verify_contract(d.read(root/'source_contract.json'),d.read(d.SOURCE))
 return model,dict(status='VERIFIED',checkpoint=c['parent_checkpoint'],ancestors=[],execution_commit=c['parent_execution_commit'],training_rx=resolved['training_rx'],heldRX_excluded=True,source_roles='EXACT_MATCH')

@torch.no_grad()
def extract(model,data,actions,device):
 zs=[];hs=[]
 for kind,a in [(None,None),*actions]:
  z=[];h=[]
  for begin in range(0,len(data['x']),256):
   x=data['x'][begin:begin+256].to(device)
   if kind:x=apply(x,kind,a,0)
   v=model.forward_identity_only(x);z.append(F.normalize(v['id_feat_joint'],dim=1,eps=1e-8).cpu());h.append(v['tx_logits'].cpu())
  zs.append(torch.cat(z));hs.append(torch.cat(h))
 return zs,hs

def run(rowid,device='cuda:0'):
 c=d.config(next(r for r in d.rows() if r['row_id']==rowid));d.validate(c);torch.set_num_threads(2);out=Path(c['output']);out.mkdir(parents=True,exist_ok=False);start=time.perf_counter()
 L,U,V,contract=load_source(c['parent_config']['source_loader_config'])
 if U is not None:raise ValueError('U must remain closed')
 train=subset(L,L['rx']!=c['held_rx']);held=subset(V,V['rx']==c['held_rx'])
 if len(train['ids'])!=5040 or len(held['ids'])!=5400:raise ValueError('Metric source holdout differs')
 with numerical_context(d.FULL_FP32_POLICY):
  model,proof=parent(c,device);before={k:v.clone() for k,v in model.state_dict().items()}
  z,unused=extract(model,train,c['actions'],device);delta=torch.stack([v-z[0] for v in z[1:]]);state=fit(z[0],train['y'],delta,c['shrinkage'],c['epsilon_relative'],c['hard_energy_fraction'])
  q,h=extract(model,held,[],device);pred={arm:(h[0].double() if arm=='NATIVE_HEAD' else logits(q[0],state,arm)) for arm in d.ARMS}
  if any(not torch.isfinite(v).all() for v in pred.values()):raise FloatingPointError('Finite metric scores')
  if any(not torch.equal(v,model.state_dict()[k]) for k,v in before.items()):raise ValueError('Fixed identity checkpoint changed')
  torch.save(dict(readouts=pred,ids=held['ids'],target_access=False),out/'source_predictions.pt');torch.save(state,out/'metric_state.pt')
  results={arm:score(v,held['y'],held['rx'],held['day']) for arm,v in pred.items()}
  done=dict(status='SOURCE_DIAGNOSTIC_COMPLETE',config=c,records=results,proof=proof,identity_updates=0,head_updates=0,target_access=False,U_access=False,parameters_buffers_unchanged=True,fit_L_count=5040,heldV_count=5400,finite_covariance_from='same-packet actual deltas; heldRX andV excluded',metric_diagnostics={k:state[k] for k in ('finite_trace','noise_trace','hard_rank','minimum_noise_eigenvalue')},elapsed_seconds=time.perf_counter()-start)
  d.write(out/'resolved_config.json',dict(config=c,pid=os.getpid(),cwd=os.getcwd(),python=sys.executable,proof=proof));d.write(out/'completion.json',done)
  print('SOURCE_DIAGNOSTIC_COMPLETE '+json.dumps(dict(row_id=rowid,metrics={arm:v['accuracy'] for arm,v in results.items()})),flush=True)
 return done
