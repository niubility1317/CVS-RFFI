import json,time,os,sys
from pathlib import Path
import torch
from torch.nn import functional as F
from . import design as d
from experiments.cvs_finite_margin import run as training
from experiments.cvs_finite_margin.run import subset
from experiments.cvs_nocfo_foundation.source import load_source
from experiments.cvs_nocfo_foundation.model import build
from experiments.cvs_soft_finite_metric.run import extract
from experiments.cvs_wave_geometry.run import score
from experiments.cvs_equivariant_identity.precision import numerical_context

def run(rowid,device='cuda:0'):
 c=d.config(next(r for r in d.rows() if r['row_id']==rowid));d.validate(c);training.d=d
 trained=training.run(rowid,device);source=Path(c['output']);out=source.parent
 if trained['identity_updates']!=4440 or trained['target_access'] or trained['U_opened'] or trained['checkpoint_sources']:raise ValueError('Prototype source training origin differs')
 L,U,V,contract=load_source(c['source_loader_config']);train=subset(L,L['rx']!=c['held_rx']);held=subset(V,V['rx']==c['held_rx'])
 ck=torch.load(source/'identity.pth',map_location='cpu',weights_only=False)
 if ck['config']!=c or ck['ancestors'] or ck['epoch']!=20 or ck['updates']!=4440 or c['held_rx'] in ck['training_rx']:raise ValueError('Own confirmation identity checkpoint differs')
 with numerical_context(d.FULL_FP32_POLICY):
  model=build(c['source_loader_config'],device);model.load_state_dict(ck['model'],strict=True);model.eval().requires_grad_(False);before={k:t.clone() for k,t in model.state_dict().items()}
  z,_=extract(model,train,[],device);q,native=extract(model,held,[],device);mu=torch.stack([z[0][train['y']==k].mean(0) for k in range(6)]);proto=F.normalize(q[0],dim=1)@F.normalize(mu,dim=1).T
  if any(not torch.equal(t,model.state_dict()[k]) for k,t in before.items()):raise ValueError('Frozen prototype model changed')
  torch.save(dict(native=native[0],prototype=proto,ids=held['ids'],target_access=False),out/'source_predictions.pt');torch.save(dict(prototypes=mu,classes=list(range(6)),fit_ids=train['ids'],fit_rx=train['rx'].unique().tolist(),held_rx=c['held_rx'],model_epoch=20),out/'prototypes.pt')
  results=dict(NATIVE_HEAD=score(native[0],held['y'],held['rx'],held['day']),COSINE_PROTO=score(proto,held['y'],held['rx'],held['day']))
  if abs(results['NATIVE_HEAD']['accuracy']-trained['metrics']['V_views'][0]['accuracy'])>1e-7:raise ValueError('Reloaded identity scores differ from frozen original')
  done=dict(status='SOURCE_DIAGNOSTIC_COMPLETE',config=c,records=results,identity_updates=4440,head_updates=0,target_access=False,U_access=False,source_training_complete=True,parameters_buffers_unchanged=True,fit_L_count=5040,heldV_count=5400,source_commit=(d.ROOT/'release_commit.txt').read_text().strip())
  d.write(out/'resolved_config.json',dict(config=c,pid=os.getpid(),cwd=os.getcwd(),python=sys.executable,source_commit=done['source_commit'],checkpoint_loaded='own same-row source finalE20 only',ancestors=[]));d.write(out/'completion.json',done)
  print('SOURCE_DIAGNOSTIC_COMPLETE '+json.dumps(dict(row_id=rowid,accuracies={a:v['accuracy'] for a,v in results.items()})),flush=True)
 return done
