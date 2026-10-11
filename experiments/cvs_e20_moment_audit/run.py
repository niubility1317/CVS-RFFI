import argparse,json,os,sys,time,copy
from pathlib import Path
import torch
from . import design as d
from experiments.cvs_e20_peer_soft.model import build
from experiments.cvs_e20_peer_soft.warmup import head_warmup
from experiments.cvs_e20_peer_soft.evaluate import snapshot,source_provenance
from experiments.cvs_e20_routed_fishr.multilevel import MultiLevel
from experiments.cvs_stage2_decouple.source import seed_all
from experiments.cvs_equivariant_identity.precision import numerical_context,actual_flags

def load_L(c):
 from experiments.cvs_phase1_stack.recover import verify_contract
 pc=c['parent_build_config'];prepared=Path(pc['preparation']);done=d.read(prepared/'preparation_complete.json')
 if done['target_access'] or done['identity_updates'] or done['status']!='SOURCE_GEOMETRY_AUDIT_COMPLETE':raise ValueError('Source preparation differs')
 packed=torch.load(prepared/'source_cache.pt',map_location='cpu',weights_only=False)
 contract=packed['contract'];verify_contract(contract,d.read(d.parent.SOURCE));verify_contract(done['source_contract'],contract)
 L={k:packed['labeled'][k] for k in ('x','y','rx','day','ids')}
 if len(L['ids'])!=6300 or len(set(L['ids']))!=6300 or sorted(L['ids'])!=sorted(contract['role_ids']['L_s']):raise ValueError('L physical identities differ')
 if set(L['ids'])&(set(contract['role_ids']['U_s'])|set(contract['role_ids']['V'])):raise ValueError('L overlaps source roles')
 del packed
 return L,contract

@torch.no_grad()
def maps_for_L(model,L,device):
 model.eval();values={k:[] for k in ('time_down','t2.pool')}
 for start in range(0,len(L['ids']),128):
  with model.multilevel.capture() as (maps,_):model(L['x'][start:start+128].to(device))
  for level in values:values[level].append(maps[level].detach().cpu())
 return {k:torch.cat(v) for k,v in values.items()}

def run(row,device='cuda:0'):
 from .fishr_stats import fishr_audit
 from .style_stats import style_audit
 c=d.validate(d.config(next(r for r in d.rows() if r['row_id']==row)));out=Path(c['output']);out.mkdir(parents=True,exist_ok=False)
 started=time.perf_counter();torch.set_num_threads(2);seed_all(c['model_seed']);cuda=str(device).startswith('cuda')
 with numerical_context(d.parent.FULL_FP32_POLICY):
  L,contract=load_L(c);pc=c['parent_build_config'];assert snapshot(pc['row_id'])[0]==pc
  endpoint=source_provenance(pc)/'final_ssdg.pth'
  resolved=dict(c,pid=os.getpid(),cwd=os.getcwd(),python=sys.executable,commit=(d.ROOT/'release_commit.txt').read_text().strip(),
   cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),hardware=torch.cuda.get_device_name(device) if cuda else 'CPU',backend_flags=actual_flags(),
   source_storage_note='Legacy packed source_cache deserializes L/V together; V discarded without forward, loss, statistic fitting or updates. U cache never loaded.',V_backward=False,U_truth=False)
  d.write(out/'resolved_config.json',resolved);d.write(out/'source_contract.json',contract);print('RESOLVED '+json.dumps(resolved),flush=True)
  models=[];peaks=[]
  for name in c['models']:
   mroot=out/name;mroot.mkdir();model=build(pc,device,checkpoint=endpoint if name=='CE20' else None)
   before={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
   model.multilevel=MultiLevel(model.base,seed=c['model_seed']).to(device)
   warm=head_warmup(dict(multilevel=c['head_plan']),model,L,device,mroot)
   model.eval();maps=maps_for_L(model,L,device);records={}
   for level in c['levels']:
    tick=time.perf_counter();head=copy.deepcopy(model.multilevel.heads[model.multilevel.key(level)]).cpu()
    fishr=fishr_audit(maps[level].mean(-1),head,L['y'],L['rx'],L['day'],L['ids'],c['model_seed'])
    style=style_audit(maps[level],L['y'],L['rx'],L['day'],L['ids'],c['model_seed'])
    records[level]=dict(fishr=fishr,style=style,seconds=time.perf_counter()-tick,shape=list(maps[level].shape))
    d.write(mroot/(level.replace('.','_')+'.json'),records[level]);print('LEVEL_COMPLETE '+json.dumps(dict(model=name,level=level,seconds=records[level]['seconds'],target_access=False)),flush=True)
   after=model.state_dict()
   if any(not torch.equal(after[k].detach().cpu(),v) for k,v in before.items()):raise ValueError('Identity state changed during moment audit')
   summary=dict(model=name,checkpoint=str(endpoint) if name=='CE20' else pc['parent_checkpoint'],parent_provenance=model._parent_provenance,
    levels=list(records),identity_state_unchanged=True,warmup=warm,identity_updates=0,L_map_readouts=6300,L_warmup_readouts=6300,
    auxiliary_head_parameters=sum(p.numel() for p in model.multilevel.heads.parameters()))
   if cuda:peaks.append(torch.cuda.max_memory_allocated(device))
   d.write(mroot/'summary.json',summary);models.append(summary)
   del model,maps,records,head
   if cuda:torch.cuda.empty_cache()
  resources=dict(seconds=time.perf_counter()-started,hardware=resolved['hardware'],peak_cuda_memory_bytes=max(peaks) if cuda else None,
   identity_updates=0,head_updates=400,L_model_packet_readouts=25200,U_forward=0,V_forward=0,target_packets=0)
  done=dict(status='SOURCE_DIAGNOSTIC_COMPLETE',config=c,models=models,resources=resources,identity_updates=0,head_updates=400,target_access=False,U_truth=False,V_backward=False,
   checkpoint_sources=c['checkpoint_sources'],artifacts=['resolved_config.json','source_contract.json','P0/summary.json','CE20/summary.json','<model>/<level>.json','<model>/head_warmup.json','<model>/head_warmup.jsonl'])
  d.write(out/'completion.json',done);print('SOURCE_DIAGNOSTIC_COMPLETE '+json.dumps(done),flush=True)
 return done

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--row',required=True);a=p.parse_args();run(a.row)
