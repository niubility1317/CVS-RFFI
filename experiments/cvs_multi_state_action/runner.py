"""Complete staged action diagnostics; source-L only, identity weights frozen."""
import argparse
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace
import torch
from experiments.cvs_multi_state_action import design as d
from experiments.cvs_multi_action_audit.runner import role_split,channel_view,save_result
from experiments.cvs_multi_disentangle.evaluate import source_provenance,checkpoint_provenance
from experiments.cvs_multi_disentangle.runtime import installed
from experiments.cvs_equivariant_identity.precision import numerical_context

def source_packets(native,c,device):
 from cvsrffi.xuc_fusion.native import role_ids_from_native
 from cvsrffi.game_tracking.data import opaque_id
 ctx=native._build_ssdg_wisig_data(d.parent.make_args(c['parent_config'],str(device)),device)
 expected=d.read(d.SOURCE)
 if ctx['named_test_loaders'] or role_ids_from_native(ctx)!=expected['role_ids']:
  raise ValueError('Actual source roles differ or target loader exists')
 ds=ctx['train_loader'].dataset
 records=[dict(id=opaque_id(v),index=i,role='L_s',y=int(v.tx_i),rx=int(v.rx_i),day=int(v.day_i)) for i,v in enumerate(ds.index)]
 fit,audit=role_split(records,expected['role_ids']['L_s'],64,32,20261009)
 def load(rows):
  vals=[ds[r['index']] for r in rows]
  return dict(x=torch.stack([v[0] for v in vals]).to(device),
   y=torch.tensor([int(v[1]) for v in vals],device=device),
   rx=torch.tensor([r['rx'] for r in rows],device=device),
   day=torch.tensor([r['day'] for r in rows],device=device),
   ids=[r['id'] for r in rows],condition=['clean']*len(rows))
 return load(fit),load(audit),dict(fit=fit,audit=audit,source_roles='EXACT_MATCH',source_contract=d.SOURCE,
  U_labels_read=False,V_read=False,target_read=False,audit_scope='unseen by fresh auxiliaries; identity has seen source L')

def joint(data):
 leo=channel_view(data)
 return {k:torch.cat([v,leo[k]],0) if torch.is_tensor(v) else
  (['clean']*len(v)+['source_practical_mid']*len(v) if k=='condition' else v+leo[k]) for k,v in data.items()}

def subset(data,mask):
 ix=torch.where(mask)[0].tolist()
 return {k:v[mask] if torch.is_tensor(v) else [v[i] for i in ix] for k,v in data.items()}

def composition_indices(data):
 selected={}
 for i,(y,rx,day,condition) in enumerate(zip(data['y'].tolist(),data['rx'].tolist(),data['day'].tolist(),data['condition'])):
  selected.setdefault((int(y),int(rx),int(day),condition),i)
 return selected

def run(c,smoke=False):
 d.validate_audit(c);torch.set_num_threads(2);device=torch.device('cpu' if smoke else 'cuda:0')
 torch.manual_seed(c['model_seed']);started=time.perf_counter()
 source=source_provenance(d.parent,c['parent_config'])
 ck=torch.load(source/'final_ssdg.pth',map_location='cpu',weights_only=False)
 checkpoint_provenance(d.parent,c['parent_config'],ck)
 with numerical_context(d.FULL_FP32_POLICY),installed(c['parent_config'],training=False) as native:
  model=native.build_baseline_model(SimpleNamespace(**ck['baseline_args']),device)
  model.load_state_dict(ck['model'],strict=True);model.eval();del ck
  with torch.no_grad():z=model(torch.zeros(2,2,256,device=device))
  if z.shape!=(2,6) or not torch.isfinite(z).all():raise ValueError('Checkpoint smoke failed')
  if smoke:return dict(status='VERIFIED',row_id=c['row_id'],checkpoint=c['checkpoint'],target_read=False)
  out=Path(c['output_root']);out.mkdir(parents=True,exist_ok=False)
  d.write(out/'resolved_config.json',dict(c,pid=os.getpid(),cwd=os.getcwd(),python=sys.executable,
   gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),hardware=torch.cuda.get_device_name(),commit=(d.ROOT/'release_commit.txt').read_text().strip()))
  d.write(out/'provenance.json',dict(status='VERIFIED',checkpoint=c['checkpoint'],
   ancestors='parent scratch exact source contract',selection='all4 fixedE200; no target scores',identity_training=False,target_access=False))
  fit,audit,roles=source_packets(native,c,device);d.write(out/'physical_roles.json',roles)
  ident=model.id_backbone;ident.eval();saved={k:v.clone() for k,v in ident.state_dict().items()}
  for p in ident.parameters():p.requires_grad_(False)
  from experiments.cvs_multi_state_action.actions import fit_and_audit,composition_audit
  from experiments.cvs_multi_state_action.receiver import receiver_audit
  f,a=joint(fit),joint(audit)
  runs=[]
  def progress(row):
   if row.get('step',0)==1 or row.get('step',0)%25==0:print('ACTION '+json.dumps(row),flush=True)
  for exposure in ('clean','source_practical_mid','joint'):
   ff,aa=(f,a) if exposure=='joint' else (subset(f,torch.tensor([v==exposure for v in f['condition']],device=device)),subset(a,torch.tensor([v==exposure for v in a['condition']],device=device)))
   print('FIT_CONFIG '+json.dumps(dict(exposure=exposure,recipe=d.AUDIT,fit=len(ff['ids']),audit=len(aa['ids']))),flush=True)
   result=fit_and_audit(ident,ff,aa,torch.Generator().manual_seed(d.AUDIT['evaluation_seed']),
    steps=200,batch_size=32,interventions=3,modes=d.AUDIT['modes'],on_step=progress)
   selected=composition_indices(aa)
   comp=dict(conditions={},selection='one fixed audit physical packet per TX/RX/day/condition',
    physical_ids=[aa['ids'][i] for i in selected.values()],strata=[list(k) for k in selected])
   for condition in sorted(set(aa['condition'])):
    ix=[i for k,i in selected.items() if k[3]==condition]
    comp['conditions'][condition]=composition_audit(ident,aa['x'][ix],torch.Generator().manual_seed(d.AUDIT['evaluation_seed']+2),result['state_dicts'],result['model_configs'])
   d.write(out/(exposure+'_composition.json'),comp)
   save_result(out,exposure+'_actions',result)
   rr=receiver_audit(ident,ff,aa,torch.Generator().manual_seed(d.AUDIT['evaluation_seed']+1),steps=200)
   save_result(out,exposure+'_receiver',rr);runs.append(exposure)
   print('EXPOSURE_COMPLETE '+exposure,flush=True)
  for tx in d.AUDIT['heldout_tx']:
   result=fit_and_audit(ident,subset(f,f['y']!=tx),a,torch.Generator().manual_seed(d.AUDIT['evaluation_seed']),
    steps=200,batch_size=32,interventions=3,heldout_tx=tx,modes=d.AUDIT['heldout_modes'],on_step=progress)
   save_result(out,'heldout_tx'+str(tx)+'_actions',result)
   print('TX_HOLDOUT_COMPLETE '+str(tx),flush=True)
  if any(not torch.equal(v,ident.state_dict()[k]) for k,v in saved.items()):raise ValueError('Identity changed')
  d.write(out/'completion.json',dict(status='SOURCE_AUDIT_COMPLETE',row_id=c['row_id'],exposures=runs,
   tx_holdouts=d.AUDIT['heldout_tx'],identity_unchanged=True,target_access=False,
   elapsed_seconds=time.perf_counter()-started,peak_cuda_bytes=torch.cuda.max_memory_allocated()))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--row',required=True);p.add_argument('--smoke',action='store_true');a=p.parse_args()
 print(run(d.audit_config(next(r for r in d.audit_rows() if r['row_id']==a.row)),a.smoke))
