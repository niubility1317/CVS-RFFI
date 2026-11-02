"""Score immutable vote-router predictions through the strict Phase1 scorer."""
import argparse,hashlib,json,shutil
from pathlib import Path
import numpy as np
from source_pilot import core
SOURCE='20261102-phase1-vote-router-ground-test-s2026092701-r01'
PREDICTION_COMMIT='8c20a392dcd6347db5f1d135b689711fd4c00e42'

def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--commit',required=True);a=p.parse_args()
 if a.out.exists():raise FileExistsError('Immutable score adapter output required')
 source=a.root/'runs'/SOURCE;sp=source/'flow/test_prediction';manifest=core.read(sp/'complete.json');audit=core.read(source/'pretruth_audit.json');keys=('status','count','views','F_run','F_row','commit','TX_truth_reads','model_updates');projected={k:manifest[k] for k in keys};core.validate_prediction_manifest(projected,count=168000,commit=PREDICTION_COMMIT)
 if audit.get('status')!='PASS' or audit.get('TX_truth_reads')!=0 or audit.get('model_updates')!=0 or set(audit['views'])!=set(core.VIEWS) or not audit.get('all_routes_recomputed_from_frozen_model'):raise ValueError('Independent pre-truth route audit required')
 ids=np.load(sp/'physical_ids.npy',allow_pickle=False)
 if ids.shape!=(168000,) or len(set(ids.tolist()))!=len(ids):raise ValueError('Unique Phase1 IDs required')
 for view in core.VIEWS:
  for arm in ('raw','restored'):
   path=sp/(view+'.'+arm+'.npy');x=np.load(path,mmap_mode='r',allow_pickle=False)
   if x.shape!=(len(ids),) or x.dtype!=np.int64 or digest(path)!=audit['views'][view][arm+'_sha256']:raise ValueError('Independently fixed prediction hash required')
 a.out.mkdir(parents=True);dst=a.out/'flow/test_prediction';dst.mkdir(parents=True);source_files=['physical_ids.npy']+[v+'.'+arm+'.npy' for v in core.VIEWS for arm in ('raw','restored')]
 for name in source_files:
  source_file=sp/name;target=dst/name
  with source_file.open('rb') as fin,target.open('xb') as fout:shutil.copyfileobj(fin,fout)
  if digest(source_file)!=digest(target):raise ValueError('Copied prediction hash differs')
 core.write(dst/'complete.json',projected);projection=dict(status='PROJECTION_FIXED',adapter_commit=a.commit,source_run=SOURCE,source_manifest_sha256=digest(sp/'complete.json'),pretruth_audit_sha256=digest(source/'pretruth_audit.json'),prediction_commit=PREDICTION_COMMIT,arrays_copied=len(source_files),source_metadata_preserved=True,TX_truth_reads=0,model_updates=0);core.write(a.out/'projection_complete.json',projection)
 core.score(a.root,a.out,PREDICTION_COMMIT);print('VOTE_ROUTER_ADAPTED_SCORE_COMPLETE',json.dumps(projection),flush=True)
if __name__=='__main__':main()
