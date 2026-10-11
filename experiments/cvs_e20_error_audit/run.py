import argparse,json,os,sys,time
from pathlib import Path
import torch
from . import design as d
from .audit import collect_predictions,paired_readout,gradient_audit
from experiments.cvs_e20_budget_fishr import design as parent
from experiments.cvs_e20_budget_fishr.model import build
from experiments.cvs_e20_budget_fishr.source import load_source
from experiments.cvs_e20_budget_fishr.warmup import head_warmup
from experiments.cvs_e20_budget_fishr.evaluate import source_provenance,snapshot
from experiments.cvs_stage2_decouple.source import seed_all
from experiments.cvs_equivariant_identity.precision import numerical_context,actual_flags


def run(row,device='cuda:0'):
 c=d.config(next(r for r in d.rows() if r['row_id']==row));d.validate(c)
 out=Path(c['output']);out.mkdir(parents=True,exist_ok=False);started=time.perf_counter()
 torch.set_num_threads(2);seed_all(c['model_seed']);cuda=str(device).startswith('cuda')
 with numerical_context(parent.FULL_FP32_POLICY):
  L,U,V,contract=load_source(c['parent_build_config']);model=build(c['parent_build_config'],device)
  resolved=dict(c,pid=os.getpid(),cwd=os.getcwd(),python=sys.executable,commit=(d.ROOT/'release_commit.txt').read_text().strip(),
   cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),hardware=torch.cuda.get_device_name(device) if cuda else 'CPU',
   backend_flags=actual_flags(),parent_provenance=model._parent_provenance,V_backward=False,U_truth=False)
  d.write(out/'resolved_config.json',resolved);d.write(out/'source_contract.json',contract)
  print('RESOLVED '+json.dumps(resolved),flush=True)
  if cuda:torch.cuda.reset_peak_memory_stats(device)
  prediction={role:dict(parent=collect_predictions(model,cache,device,c['batch_size'],role=role),endpoints={}) for role,cache in [('L',L),('V',V)]}
  warmroot=out/'head_probe';warmroot.mkdir();warm=head_warmup(c['parent_build_config'],model,L,device,warmroot)
  gradient=gradient_audit(model,L,U,c['parent_build_config'],rounds=c['gradient_rounds'],device=device)
  d.write(out/'gradient_audit.json',gradient)
  del model
  if cuda:torch.cuda.empty_cache()
  paired=[];sources=[]
  for arm in c['endpoint_arms']:
   ec=parent.config(next(r for r in parent.rows() if r['arm']==arm and r['model_seed']==c['model_seed']))
   assert snapshot(ec['row_id'])[0]==ec
   source=source_provenance(ec);checkpoint=source/'final_ssdg.pth';end=build(ec,device,checkpoint=checkpoint)
   for role,cache in [('L',L),('V',V)]:
    value=collect_predictions(end,cache,device,c['batch_size'],role=role)
    prediction[role]['endpoints'][arm]=value
    paired.append(dict(arm=arm,role=role,result=paired_readout(prediction[role]['parent'],value)))
   sources.append(dict(arm=arm,checkpoint=str(checkpoint),parent=ec['parent_checkpoint'],selection='all fixed arms / final incremental20',source_provenance_verified=True))
   print('READOUT_COMPLETE '+json.dumps(dict(arm=arm,L=6300,V=27000,V_backward=False,target_access=False)),flush=True)
   del end
   if cuda:torch.cuda.empty_cache()
  versus_ce=[dict(arm=arm,role=role,result=paired_readout(values['endpoints']['CE'],values['endpoints'][arm]))
             for role,values in prediction.items() for arm in c['endpoint_arms'] if arm!='CE']
  d.write(out/'paired_readouts.json',dict(rows=paired,versus_CE=versus_ce,confidence=.95,cohorts_frozen_from_parent=True,
   versus_CE_cohorts='separately defined from fixed CE endpoint',V_backward=False,target_access=False))
  torch.save(prediction,out/'source_predictions.pt')
  resources=dict(seconds=time.perf_counter()-started,hardware=resolved['hardware'],peak_cuda_memory_bytes=torch.cuda.max_memory_allocated(device) if cuda else None,
   identity_updates=0,head_updates=warm['optimizer_steps'],source_readout_packets=9*(6300+27000),target_packets=0,gradient_rounds=24,objectives_per_round=8)
  done=dict(status='SOURCE_DIAGNOSTIC_COMPLETE',config=c,identity_updates=0,head_warmup=warm,resources=resources,endpoint_sources=sources,
   checkpoint_sources=[c['source_checkpoint']]+[s['checkpoint'] for s in sources],target_access=False,V_backward=False,U_truth=False,
   artifacts=['resolved_config.json','source_contract.json','source_predictions.pt','paired_readouts.json','gradient_audit.json','head_probe/head_warmup.json','head_probe/head_warmup.jsonl'])
  d.write(out/'completion.json',done);print('SOURCE_DIAGNOSTIC_COMPLETE '+json.dumps(done),flush=True)
 return done


if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--row',required=True);args=p.parse_args();run(args.row)
