import argparse,os,sys,time,json
from pathlib import Path
import torch
from . import design as d
from .anchors import fit,predict
from experiments.cvs_stage2_teacher_factorial.source import load_source
from experiments.cvs_stage2_teacher_factorial.model import build
from experiments.cvs_source_anchor_audit.run import extract,metrics
from experiments.cvs_equivariant_identity.precision import numerical_context,actual_flags

def run(row_id,device='cuda:0'):
 c=d.config(next(r for r in d.rows() if r['row_id']==row_id));d.validate(c);out=Path(c['output']);out.mkdir(parents=True,exist_ok=False)
 started=time.perf_counter();torch.set_num_threads(2);L,U,V,contract,provenance=load_source(c['parent_build_config'])
 d.write(out/'resolved_config.json',dict(config=c,pid=os.getpid(),cwd=os.getcwd(),python=sys.executable,commit=(d.ROOT/'release_commit.txt').read_text().strip(),CUDA_VISIBLE_DEVICES=os.environ.get('CUDA_VISIBLE_DEVICES'),target_loaders_constructed=False))
 d.write(out/'provenance.json',dict(parent=provenance,loaded_checkpoint=c['source_checkpoint'],E40_loaded=False,source_contract=contract))
 with numerical_context(d.parent.FULL_FP32_POLICY):
  model=build(c['parent_build_config'],device).eval();initial={n:t.detach().clone() for n,t in model.state_dict().items()}
  if str(device).startswith('cuda'):torch.cuda.reset_peak_memory_stats()
  lf,lq=extract(model,L,device);uf,uq=extract(model,U,device)
  states={mode:fit(lf,L['y'].to(device),L['rx'].to(device),c['anchor'],mode) for mode in c['anchor']['metric_modes']}
  d.write(out/'anchor_fit.json',{mode:s['diagnostics'] for mode,s in states.items()})
  ureport={}
  for mode,s in states.items():
   _,diag=predict(s,uf,U['rx'].to(device),uq);ureport[mode]=diag
  d.write(out/'U_diagnostics.json',dict(metrics=ureport,truth_hidden=True,labels_used=False))
  # All transformations and anchors are fixed before V is extracted.
  vf,vq=extract(model,V,device);report={}
  for mode,s in states.items():
   probs,diag=predict(s,vf,V['rx'].to(device),vq);report[mode]=dict(readouts=metrics(probs,V['y'].to(device),V['rx'].to(device),V['day'].to(device)),diagnostics=diag)
  d.write(out/'V_readouts.json',dict(metric_modes=report,role='V',V_fit=False,target_access=False,parent_already_saw_all_source_RX=True))
  assert all(torch.equal(initial[n],t) for n,t in model.state_dict().items())
  done=dict(status='SOURCE_DIAGNOSTIC_COMPLETE',row_id=row_id,config=c,identity_updates=0,target_access=False,U_truth_access=False,V_in_fit=False,source_model_unchanged=True,elapsed_seconds=time.perf_counter()-started,peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated() if str(device).startswith('cuda') else None,backend=actual_flags())
  d.write(out/'completion.json',done);print(json.dumps(done),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--row',required=True);a=p.parse_args();run(a.row)
