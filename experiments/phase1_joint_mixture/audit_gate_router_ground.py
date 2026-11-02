"""Verify every fixed ground route before any ground truth is opened."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import torch
from gate_aware_router import GateAwareRouter

NAMES=['raw','physical','unroll025','unroll05','unroll1'];VIEWS=['clean','practical_high','practical_mid','practical_low_suburban','practical_high_urban','practical_mid_urban','practical_low_urban'];SOURCE='20261102-phase1-gate-aware-router-source-s2026092701-r01';SOURCE_COMMIT='9dac8eaa0a35e352652791bf0080ade59751af09';PRIOR_SHA='fa90fd400bed527ca6f0409ccd09a0f3ad6a979492f77e8fcf35fdf7d0b6d6a5'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def verify_block(net,candidates,gate,route,raw,restored):
 n=len(gate)
 if candidates.shape!=(n,5) or candidates.dtype!=np.int64 or gate.shape!=(n,) or gate.dtype!=np.float32 or route.shape!=(n,) or route.dtype!=np.uint8 or raw.shape!=(n,) or restored.shape!=(n,) or raw.dtype!=np.int64 or restored.dtype!=np.int64 or not np.isfinite(gate).all() or np.any((gate<0)|(gate>1)) or np.any((candidates<0)|(candidates>=6)) or np.any(route>=5):raise ValueError('Valid fixed candidate and route arrays required')
 if not np.array_equal(candidates[:,0],raw) or not np.array_equal(candidates[gate<=.8],np.broadcast_to(raw[gate<=.8,None],candidates[gate<=.8].shape)) or not np.array_equal(restored,candidates[np.arange(n),route]):raise ValueError('Fixed candidate selection binding failed')
 with torch.no_grad():choice,pred=net.choose(torch.tensor(candidates.tolist(),dtype=torch.long),torch.tensor(gate.tolist(),dtype=torch.float32))
 if not np.array_equal(np.from_dlpack(choice.contiguous()),route.astype(np.int64)) or not np.array_equal(np.from_dlpack(pred.contiguous()),restored):raise ValueError('Independent fixed router replay failed')
 return n

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--commit',required=True);a=ap.parse_args();result_path=a.out/'pretruth_route_audit.json'
 if result_path.exists():raise FileExistsError(result_path)
 torch.set_num_threads(8);source=a.root/'runs'/SOURCE;m=json.loads((source/'result.json').read_text());ckpath=source/'train46_eval8.pt';ck=torch.load(ckpath,map_location='cpu',weights_only=False)
 if m.get('status')!='SOURCE_GATE_ROUTER_TRAINING_COMPLETE' or m.get('commit')!=SOURCE_COMMIT or m.get('prior_sha256')!=PRIOR_SHA or m.get('fold_model_sha256',{}).get('train46_eval8')!=sha(ckpath) or ck.get('commit')!=SOURCE_COMMIT or ck.get('prior_sha256')!=PRIOR_SHA or ck.get('candidate_names')!=NAMES or ck.get('fold',{}).get('train_receivers')!=[4,6] or ck.get('fold',{}).get('eval_receivers')!=[8]:raise ValueError('Exact fixed source fold required')
 net=GateAwareRouter();net.load_state_dict(ck['model'],strict=True);net.eval().requires_grad_(False);dst=a.out/'flow/test_prediction';manifest=json.loads((dst/'pending_complete.json').read_text());baseline=a.root/'runs/20261101-phase1-dura-trisigtime-baseline-s2026092701-r01/flow/test_prediction';ids=np.load(dst/'physical_ids.npy',allow_pickle=False)
 if manifest.get('status')!='PREDICTIONS_COMPLETE' or manifest.get('commit')!=a.commit or manifest.get('count')!=168000 or manifest.get('views')!=VIEWS or manifest.get('TX_truth_reads')!=0 or manifest.get('model_updates')!=0 or ids.shape!=(168000,) or len(np.unique(ids))!=len(ids) or not np.array_equal(ids,np.load(baseline/'physical_ids.npy',allow_pickle=False)):raise ValueError('Fixed pending manifest and ground IDs required')
 checks={}
 for view in VIEWS:
  paths={k:dst/(view+'.'+k+'.npy') for k in ('raw','restored','candidates','gate','route')};arr={k:np.load(p,mmap_mode='r',allow_pickle=False) for k,p in paths.items()};ref=np.load(baseline/(view+'.raw.npy'),mmap_mode='r',allow_pickle=False)
  if not np.array_equal(arr['raw'],ref):raise ValueError('Ground raw baseline differs')
  checked=0
  for start in range(0,len(ids),32):
   q=slice(start,start+32);checked+=verify_block(net,arr['candidates'][q],arr['gate'][q],arr['route'][q],arr['raw'][q],arr['restored'][q])
  if checked!=len(ids):raise ValueError('All source candidate routes must be checked')
  checks[view]=dict(count=checked,sha256={k:sha(p) for k,p in paths.items()})
 out=dict(status='PASS',count=len(ids),views=checks,checkpoint_sha256=sha(ckpath),all_routes_recomputed_from_frozen_model=True,all_selected_decisions_match_stored_candidates=True,ground_candidate_IQ_independently_recomputed=False,TX_truth_reads=0,U_s_truth_reads=0,V_reads=0,test_IQ_reads=0,model_updates=0,caveat='Independent audit replays selection from stored candidate decisions; candidate IQ itself was checked by prediction program, not independently regenerated.')
 with result_path.open('x',encoding='utf8') as f:json.dump(out,f,indent=2);f.write('\n')
 print(json.dumps({'status':'PASS','views':len(checks),'records_per_view':len(ids)}),flush=True)
if __name__=='__main__':main()
