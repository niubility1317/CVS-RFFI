import json
from pathlib import Path
from . import plan as p

def read(path):return json.loads(Path(path).read_text())
def write(path,value):Path(path).write_text(json.dumps(value,allow_nan=False)+'\n')

def native_source_logits(native,x):
 import torch
 # Original centered-Sinc needs ordinary versioned intermediates for its cache.
 with torch.inference_mode(False),torch.no_grad():
  return native.forward_identity_only(x)['tx_logits']

def observations():
 import numpy as np
 from experiments.cvs_t1_e50_confirm.evaluate import observations as original
 from experiments.cvs_t1_e50_confirm import design as d
 ids,manifest=original()
 done=read(p.DURA_PRED/'complete.json');source=read(p.DURA_ROW/'checkpoint_source.json')
 required=dict(status='PREDICTIONS_COMPLETE',kind='native_map_prior',role='target',count=p.COUNT,views=list(p.VIEWS),
  TX_truth_reads=0,state_updates=0,head_fit_updates=0,original_default_head=True,execution='stage2_residual_graph',batch=128,native_fallback=False)
 if any(done.get(k)!=v for k,v in required.items()):raise ValueError('Complete exact DURA v1 graph/RR outputs required')
 if (p.DURA_RELEASE/'release_commit.txt').read_text().strip()!=p.DURA_COMMIT or source['path']!=str(p.DURA_CHECKPOINT) or source['source_arm']!='stage2' or source['epoch']!=20 or source['updates']!=4940 or source['training_F_access'] or source['new_training_updates']:
  raise ValueError('DURA v1 frozen source/runtime differs')
 if read(p.DURA_ROW/'complete.json')['status']!='ANALYZED' or not read(p.DURA_ROW/'smoke.json')['original_weights_unchanged']:
  raise ValueError('Original complete DURA artifact required')
 if not np.array_equal(np.load(p.DURA_PRED/'physical_ids.npy',allow_pickle=False),ids):raise ValueError('DURA/current physical ordering differs')
 raw={v:Path(manifest['clean_ref']) if v=='clean' else d.FORMAL_VIEWS/(v+'.npy') for v in p.VIEWS}
 restored={v:p.DURA_PRED/(v+'.restored.iq.npy') for v in p.VIEWS}
 for path in (*raw.values(),*restored.values()):
  a=np.load(path,mmap_mode='r',allow_pickle=False)
  if a.shape!=(p.COUNT,2,256) or a.dtype!=np.float32:raise ValueError('Fixed current/restored IQ shape differs')
 return ids,raw,restored

def load_reader(r,device):
 import torch
 from experiments.pamf_fcrossrx_readout.frozen import load_reader as load_noas,from_state
 from experiments.pamf_restoration.frozen import check_contract
 src=p.source(r);check_contract(src,p.SOURCE_CACHE)
 if r['arm']=='NOAS':return load_noas(src,p.SOURCE_CACHE,device,seed=r['seed']),None
 if r['seed']==p.SEEDS[0]:
  from experiments.cvs_t1_e50 import design as d,model as m
 else:
  from experiments.cvs_t1_e50_confirm import design as d,model as m
 c=d.config(next(x for x in d.rows() if x['row_id']=='E30PLUS20-s'+str(r['seed'])))
 native=m.own_model(c,device).eval().requires_grad_(False)
 payload=torch.load(src/'final_ssdg.pth',map_location='cpu',weights_only=False)
 if payload['args']['seed']!=r['seed'] or c['cumulative_epochs']!=50 or c['cumulative_updates']!=11100 or c['temperature']['As_KL']!=0 or payload['epoch']!=20:
  raise ValueError('Exact same-seed original SARD E50 required')
 reader=from_state(payload['model'],r['seed']).to(device)
 return reader,native

def smoke(r,reader,native,device):
 import numpy as np,torch
 from experiments.pamf_r46_multipoint_fast.tensor_io import array_tensor
 path=p.PROJECT/'runs/20261019-phase1-restoration-learned-action-source-s2026092701-r07/cache'
 roles=np.load(path/'train/roles.npy',allow_pickle=False);indices=np.flatnonzero(roles=='L_s')[:2]
 x=array_tensor(np.load(path/'epochs/01/received.npy',mmap_mode='r',allow_pickle=False)[indices],device)
 before={k:v.clone() for k,v in reader.state_dict().items()}
 with torch.inference_mode():
  logits=reader(x)
  if logits.shape!=(2,6) or not torch.isfinite(logits).all():raise ValueError('Frozen original reader smoke failed')
  if native is not None:torch.testing.assert_close(logits,native_source_logits(native,x),atol=1e-5,rtol=2e-5)
 if any(v.requires_grad or v.grad is not None for v in reader.parameters()) or any(not torch.equal(v,reader.state_dict()[k]) for k,v in before.items()):raise ValueError('Frozen reader state changed')
 return dict(status='PASS',real_source_Ls=2,target_reads=0,updates=0,head_fit_updates=0,seed=r['seed'],
  native_defaulthead_equivalence=native is not None,NOAS_existing_strict_reader=r['arm']=='NOAS',parameter_count=sum(x.numel() for x in reader.parameters()))

def existing_predictions(r,mode,view,ids):
 import numpy as np
 if not p.reused(r,mode,view):raise ValueError('Existing decisions only for registered reused cells')
 if r['arm']=='SARD':
  folder=p.source(r).parent/'prediction';done=read(folder/'complete.json')
  if done['status']!='PREDICTIONS_COMPLETE' or done['views']!=list(p.VIEWS) or done['truth_read'] or done['head_fit_updates']:raise ValueError('SARD prior full prediction differs')
  with np.load(folder/(view+'_scores.npz'),allow_pickle=False) as f:
   if not np.array_equal(f['ids'],ids):raise ValueError('Existing SARD IDs differ')
   return f['logits'].argmax(1).astype(np.int64),str(folder/(view+'_scores.npz'))
 if r['seed']==p.SEEDS[0]:
  folder=p.RAW_F50/'prediction' if mode=='raw' else p.DURA_PRED
  done=read(folder/'complete.json')
  if done['status']!='PREDICTIONS_COMPLETE' or done['views']!=list(p.VIEWS) or done['count']!=p.COUNT or done['F_seed']!=r['seed'] or done['TX_truth_reads'] or done.get('state_updates',0):raise ValueError('Original NoAs full prediction differs')
  if not np.array_equal(np.load(folder/'physical_ids.npy',allow_pickle=False),ids):raise ValueError('Original NoAs ordering differs')
  path=folder/(view+('.identity.logits.npy' if mode=='raw' else '.restored.logits.npy'))
  return np.load(path,mmap_mode='r',allow_pickle=False).argmax(1).astype(np.int64),str(path)
 folder=p.source(r).parent/'prediction';done=read(folder/'complete.json')
 if mode!='raw' or view!='clean' or done['status']!='PREDICTIONS_COMPLETE' or done['truth_read'] or done['query_fit'] or done['head_fit_updates'] or done['config']['model_seed']!=r['seed']:raise ValueError('Existing clean-only row differs')
 with np.load(folder/'predictions.npz',allow_pickle=False) as f:
  if not np.array_equal(f['ids'],ids):raise ValueError('Original clean IDs differ')
  return f['NATIVE_HEAD'].copy(),str(folder/'predictions.npz')
