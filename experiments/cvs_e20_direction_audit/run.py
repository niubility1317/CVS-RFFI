import argparse,json,os,sys,time
from pathlib import Path
import torch
from . import design as d
from experiments.cvs_e20_cross_style.model import build
from experiments.cvs_e20_cross_style.warmup import head_warmup
from experiments.cvs_e20_cross_style.evaluate import snapshot,source_provenance
from experiments.cvs_e20_cross_style.multilevel import MultiLevel,calibrate
from experiments.cvs_e20_moment_audit.run import load_L
from experiments.cvs_stage2_decouple.source import seed_all
from experiments.cvs_equivariant_identity.precision import numerical_context,actual_flags

def run(row,device='cuda:0'):
 from .diagnostic import audit
 c=d.validate(d.config(next(r for r in d.rows() if r['row_id']==row)))
 out=Path(c['output']);out.mkdir(parents=True,exist_ok=False)
 tick=time.perf_counter();torch.set_num_threads(2);seed_all(c['model_seed']);cuda=str(device).startswith('cuda')
 with numerical_context(d.parent.FULL_FP32_POLICY):
  L,contract=load_L(c);L['role']='L_s';pc=c['parent_build_config'];assert snapshot(pc['row_id'])[0]==pc
  endpoint=source_provenance(pc)/'final_ssdg.pth'
  resolved=dict(c,pid=os.getpid(),cwd=os.getcwd(),python=sys.executable,commit=(d.ROOT/'release_commit.txt').read_text().strip(),cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),hardware=torch.cuda.get_device_name(device) if cuda else 'CPU',backend_flags=actual_flags(),source_storage_note='Packed source cache co-deserializes V then discards without forward/fitting; U cache not loaded.',V_backward=False,U_truth=False)
  d.write(out/'resolved_config.json',resolved);d.write(out/'source_contract.json',contract)
  print('RESOLVED '+json.dumps(resolved),flush=True);models=[]
  for name in c['models']:
   mroot=out/name;mroot.mkdir();model=build(pc,device,checkpoint=endpoint if name=='CE20' else None)
   before={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
   model.multilevel=MultiLevel(model.base,seed=c['model_seed']).to(device)
   if cuda:torch.cuda.reset_peak_memory_stats(device)
   warm=head_warmup(c,model,L,device,mroot);cal=calibrate(c,model,L,device,mroot)
   print('AUDIT_STARTED '+json.dumps(dict(model=name,rounds=36,styles=6)),flush=True)
   result=audit(model,L,c,device);d.write(mroot/'audit.json',result)
   if any(not torch.equal(model.state_dict()[k].detach().cpu(),v) for k,v in before.items()):raise ValueError('Original identity changed')
   summary=dict(model=name,checkpoint=str(endpoint) if name=='CE20' else pc['parent_checkpoint'],parent_provenance=model._parent_provenance,
    identity_state_unchanged=True,identity_updates=0,warmup=warm,calibration=cal,auxiliary_head_parameters=sum(p.numel() for p in model.multilevel.heads.parameters()),
    peak_cuda_memory_bytes=torch.cuda.max_memory_allocated(device) if cuda else None)
   d.write(mroot/'summary.json',summary);models.append(summary)
   print('MODEL_COMPLETE '+json.dumps(dict(model=name,identity_updates=0,target_access=False)),flush=True)
   del model,result
   if cuda:torch.cuda.empty_cache()
  done=dict(status='SOURCE_DIAGNOSTIC_COMPLETE',config=c,models=models,identity_updates=0,head_updates=400,target_access=False,U_truth=False,V_backward=False,U_forward=0,V_forward=0,
   checkpoint_sources=c['checkpoint_sources'],seconds=time.perf_counter()-tick,hardware=resolved['hardware'])
  d.write(out/'completion.json',done);print('SOURCE_DIAGNOSTIC_COMPLETE '+json.dumps(done),flush=True)
 return done

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--row',required=True);a=p.parse_args();run(a.row)
