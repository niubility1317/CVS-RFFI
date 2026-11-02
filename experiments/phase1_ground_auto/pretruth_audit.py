"""Check every fixed Phase1 ground prediction before opening any truth."""
import hashlib,json
from pathlib import Path
import numpy as np

def check(candidate,baseline,count,commit,views):
 candidate=Path(candidate);baseline=Path(baseline);cm=json.loads((candidate/'complete.json').read_text());bm=json.loads((baseline/'complete.json').read_text())
 for m in (cm,bm):
  if m.get('status')!='PREDICTIONS_COMPLETE' or m.get('count')!=count or m.get('views')!=views or m.get('TX_truth_reads')!=0 or m.get('model_updates')!=0:raise ValueError('Complete frozen seven-view predictions required')
 if cm.get('commit')!=commit or cm.get('F_run')!=bm.get('F_run') or cm.get('F_row')!=bm.get('F_row') or not cm.get('F_run') or not cm.get('F_row'):raise ValueError('Exact candidate commit and shared frozen F required')
 ids=np.load(candidate/'physical_ids.npy',allow_pickle=False);ref=np.load(baseline/'physical_ids.npy',allow_pickle=False)
 if ids.shape!=(count,) or len(np.unique(ids))!=count or not np.array_equal(ids,ref):raise ValueError('Unique identical physical IDs required')
 files=[('candidate/physical_ids.npy',candidate/'physical_ids.npy'),('baseline/physical_ids.npy',baseline/'physical_ids.npy')]
 for view in views:
  for arm in ('raw','restored'):
   p=candidate/(view+'.'+arm+'.npy');q=baseline/(view+'.'+arm+'.npy');x=np.load(p,mmap_mode='r',allow_pickle=False);y=np.load(q,mmap_mode='r',allow_pickle=False)
   if x.shape!=(count,) or y.shape!=(count,) or x.dtype!=np.int64 or y.dtype!=np.int64 or not np.all((x>=0)&(x<6)) or not np.all((y>=0)&(y<6)):raise ValueError('Complete in-range Phase1 predictions required')
   if arm=='raw' and not np.array_equal(x,y):raise ValueError('Raw frozen F predictions must match for every record')
   files.extend([('candidate/'+p.name,p),('baseline/'+q.name,q)])
 def digest(path):
  h=hashlib.sha256()
  with path.open('rb') as f:
   for chunk in iter(lambda:f.read(1048576),b''):h.update(chunk)
  return h.hexdigest()
 return dict(status='PASS',count=count,views=views,commit=commit,F_run=cm['F_run'],F_row=cm['F_row'],raw_and_physical_ids_match=True,TX_truth_reads=0,model_updates=0,sha256={name:digest(path) for name,path in files})
