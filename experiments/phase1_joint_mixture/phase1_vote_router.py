"""Evaluate a fixed source-trained vote router on fixed Phase1 candidate outputs."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import torch
from source_pilot import core
from vote_router import VoteRouter
from train_vote_router import NAMES
TRAIN='20261102-phase1-vote-router-source-s2026092701-r01'
TRAIN_COMMIT='4f065676bc23c5b1adaa9e7eab2c7dfd299e849f'
RUNS=['20261102-phase1-free-mixture-ground-test-s2026092701-r01','20261102-phase1-free-full-hard-ground-test-s2026092701-r01','20261102-phase1-hierarchical-prior-ground-test-s2026092701-r01','20261102-phase1-joint-mixture-ground-test-s2026092701-r01','20261102-phase1-confidence-router-ground-test-s2026092701-r01','20261102-phase1-attention-waveform-ground-test-s2026092701-r01','20261102-phase1-structured-prior-ground-test-s2026092701-r01','20261102-phase1-independent-prior-pnp-ground-test-s2026092701-r02']

def predict(root,out,commit):
 if out.exists():raise FileExistsError('Immutable ground output required')
 torch.set_num_threads(2);folder=root/'runs'/TRAIN;result=core.read(folder/'result.json');checkpoint=folder/'checkpoint.pt'
 if result.get('status')!='SOURCE_ROUTER_TRAINING_COMPLETE' or result.get('commit')!=TRAIN_COMMIT or result.get('candidate_names')!=NAMES or result.get('U_s_truth_reads')!=0 or result.get('test_reads')!=0 or hashlib.sha256(checkpoint.read_bytes()).hexdigest()!=result.get('checkpoint_sha256'):raise ValueError('Exact frozen source router required')
 state=torch.load(checkpoint,map_location='cpu',weights_only=False)
 if state.get('commit')!=TRAIN_COMMIT or state.get('candidate_names')!=NAMES or state.get('fit_receivers')!=[4,6]:raise ValueError('Exact source-only router checkpoint required')
 model=VoteRouter();model.load_state_dict(state['model'],strict=True);model.eval().requires_grad_(False);base=root/'runs'/RUNS[0]/'flow/test_prediction';ids=np.load(base/'physical_ids.npy',allow_pickle=False)
 if ids.shape!=(168000,) or len(set(ids.tolist()))!=len(ids):raise ValueError('Fixed unique Phase1 ground IDs required')
 candidates=[]
 for run in RUNS:
  path=root/'runs'/run/'flow/test_prediction';m=core.read(path/'complete.json')
  if m.get('status')!='PREDICTIONS_COMPLETE' or m.get('count')!=168000 or m.get('views')!=list(core.VIEWS) or m.get('TX_truth_reads')!=0 or m.get('model_updates')!=0 or not np.array_equal(np.load(path/'physical_ids.npy',allow_pickle=False),ids):raise ValueError('Fixed candidate predictions and IDs required: '+run)
  candidates.append(path)
 out.mkdir(parents=True);dst=out/'flow/test_prediction';dst.mkdir(parents=True);np.save(dst/'physical_ids.npy',ids,allow_pickle=False);route_counts={}
 with torch.no_grad():
  for view in core.VIEWS:
   raw=np.load(base/(view+'.raw.npy'),allow_pickle=False);columns=[raw]
   for path in candidates:
    candidate=np.load(path/(view+'.restored.npy'),allow_pickle=False);common=np.load(path/(view+'.raw.npy'),allow_pickle=False)
    if not np.array_equal(common,raw) or candidate.shape!=(len(ids),) or candidate.dtype!=np.int64 or np.any((candidate<0)|(candidate>=6)):raise ValueError('Same raw F and valid fixed candidate predictions required')
    columns.append(candidate)
   data=np.column_stack(columns);out_pred=np.empty(len(ids),dtype=np.int64);routes=np.empty(len(ids),dtype=np.uint8)
   for start in range(0,len(ids),2048):
    x=torch.tensor(data[start:start+2048].tolist(),dtype=torch.long);choice,pred=model.choose(x);end=start+len(x);out_pred[start:end]=np.from_dlpack(pred.contiguous());routes[start:end]=np.from_dlpack(choice.to(torch.uint8).contiguous())
   if not np.array_equal(out_pred,data[np.arange(len(ids)),routes]):raise ValueError('Selected output must be a fixed candidate prediction')
   np.save(dst/(view+'.raw.npy'),raw,allow_pickle=False);np.save(dst/(view+'.restored.npy'),out_pred,allow_pickle=False);np.save(dst/(view+'.route.npy'),routes,allow_pickle=False);route_counts[view]={NAMES[k]:int(np.count_nonzero(routes==k)) for k in range(len(NAMES))};print('VOTE_ROUTER_VIEW_COMPLETE',view,len(ids),flush=True)
 m=dict(status='PREDICTIONS_COMPLETE',count=len(ids),views=list(core.VIEWS),F_run=core.F_RUN,F_row=core.F_ROW,commit=commit,TX_truth_reads=0,model_updates=0,candidate_names=NAMES,candidate_runs=RUNS,router_checkpoint_sha256=result['checkpoint_sha256'],routes_by_view=route_counts,derived_from_fixed_candidate_predictions=True);core.write(dst/'complete.json',m);print('VOTE_ROUTER_PREDICTIONS_COMPLETE',flush=True)

def score(root,out,commit):
 dst=out/'flow/test_prediction';m=core.read(dst/'complete.json');core.validate_prediction_manifest(m,count=168000,commit=commit)
 if m.get('TX_truth_reads')!=0 or m.get('model_updates')!=0 or m.get('candidate_names')!=NAMES:raise ValueError('Frozen complete predictions required')
 for view in core.VIEWS:
  for arm in ('raw','restored'):
   a=np.load(dst/(view+'.'+arm+'.npy'),mmap_mode='r',allow_pickle=False)
   if a.shape!=(168000,) or a.dtype!=np.int64:raise ValueError('Complete fixed prediction arrays required')
 core.score(root,out,commit);print('VOTE_ROUTER_TRUTH_SCORE_COMPLETE',flush=True)

def main():
 p=argparse.ArgumentParser();p.add_argument('mode',choices=('predict','score'));p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--commit',required=True);a=p.parse_args();predict(a.root,a.out,a.commit) if a.mode=='predict' else score(a.root,a.out,a.commit)
if __name__=='__main__':main()
