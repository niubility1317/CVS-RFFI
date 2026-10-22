import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];P=ROOT/'local_artifacts/external_sat_concat_audit_20261023';sys.path.insert(0,str(ROOT))
from experiments.cric_dura_comparison.publish import ssh
rt=json.loads((ROOT/'local_artifacts/phase1_external_dg_aug_pl_gpu23_r3_20261019/runtime.json').read_text(encoding='utf-8'))
selected=json.loads((P/'selected_source_rows.json').read_text(encoding='utf-8'))
items=[dict(root=r['root'],row_id=r['config']['row_id'],config=r['config']) for r in selected]
inner=r'''
import gc,json,math,time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from experiments.phase1_external_dg_aug import source
from experiments.phase1_external_dg_aug.models import build
from experiments.phase1_external_dg_aug.objectives import step
from experiments.phase1_external_dg_aug.augmentation import DefaultConcat,default_apply
from experiments.phase1_dg_nativebudget import pipeline as native
torch.set_num_threads(2);rt=RUNTIME;root=Path(rt['run_root']);plan=json.loads(Path(rt['matrix']).read_text())
chosen=[c for c in plan['rows'] if c['family'] in ('riei','drift','ibrff','sdg') and c['group'] in ('F0','F0-pl') and c['model_seed']==2026092701]
assert len(chosen)==16
cases=[];channels=[]
for row in chosen:
 frozen=json.loads((root/row['row_id']/'source/freeze.json').read_text());c=frozen['config']
 assert all(c.get(k)==v for k,v in row.items())
 meta=json.loads((Path(c['source_cache'])/'metadata.json').read_text())
 def arrays(role):
  p=Path(meta.get('role_cache_roots',{}).get(role,c['source_cache']))
  a=np.load(p/(role+'_x.npy'),mmap_mode='r');ix=np.load(p/(role+'_index.npz'),allow_pickle=False)
  x=torch.tensor(a[:6].copy());rx=torch.tensor(ix['rx'][:6].copy(),dtype=torch.long)
  if role=='U_s':
   assert set(ix.files)=={'ids','rx'};return x,rx
  return x,torch.tensor(ix['y'][:6].copy(),dtype=torch.long),rx
 x,y,rx=arrays('L_s');u,urx=arrays('U_s') if c['U_s_use'] else (None,None)
 if not channels:
  for index,scenario in enumerate(('leo_clear_weak','leo_low_elev_weak','leo_rain_weak')):
   out,info=default_apply(x,scenario,None,gen=torch.Generator().manual_seed(2027+index),return_meta=True)
   relative=float(((out-x).double().square().sum()/x.double().square().sum()).sqrt())
   assert out.shape==x.shape and torch.isfinite(out).all() and relative>0
   channels.append(dict(scenario=scenario,finite=True,shape=list(out.shape),relative_L2_change=relative))
 for stage in range(1,c['stages']+1):
  native.seed(c['model_seed']);net=build(c)
  path=Path(frozen['checkpoint']) if stage==c['stages'] else root/row['row_id']/'source/stage1/best.pt'
  payload=torch.load(path,map_location='cpu')
  assert payload['config']==c and payload['source_contract']==frozen['source_contract']
  assert payload['training_telemetry']['satellite_applied_examples']>0
  net.load_state_dict(payload['state_dict'],strict=True);net.train()
  author=None;views=None
  if c['family']=='sdg' and c['variant']=='full':
   author=native.author_sdg(c)
   sample=source.SDGDataset(torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x,y,rx),batch_size=6),author)
   xx,yy,rr,views=next(iter(torch.utils.data.DataLoader(sample,batch_size=6)))
   assert torch.equal(xx,x) and torch.equal(yy,y) and torch.equal(rr,rx)
  base=views[-1] if views is not None else x
  aug=DefaultConcat(c['augmentation_seed'])
  for bi in range(100):
   satellite,sat_info=aug(base,.75,bi,stage)
   if sat_info['applied']:break
  assert sat_info['applied'] and not torch.equal(satellite,base)
  parts=(list(views) if views is not None else [x])+[satellite]+([u] if u is not None else [])
  expected=torch.cat(parts);sat_start=(2 if views is not None else 1)*len(y)
  calls=[];heads=[];handles=[];case={}
  entry=net.fed if c['family']=='riei' else net.encoder if c['family']=='drift' else net
  handles.append(entry.register_forward_pre_hook(lambda module,inputs:calls.append(inputs[0].detach().clone())))
  for name,module in net.named_modules():
   if isinstance(module,torch.nn.Linear) and module.out_features==c['num_classes']:
    handles.append(module.register_forward_hook(lambda module,inputs,out,name=name:heads.append((name,out))))
  original=F.cross_entropy;ce_calls=[];all_params=[(n,p) for n,p in net.named_parameters() if p.requires_grad]
  def track_ce(logits,labels,*args,**kwargs):
   loss=original(logits,labels,*args,**kwargs);ce_calls.append(loss)
   if len(ce_calls)==2:
    assert torch.equal(labels,y) and logits.shape==(len(y),c['num_classes'])
    gradients=torch.autograd.grad(loss,[p for n,p in all_params],retain_graph=True,allow_unused=True)
    nonzero={name:float(g.detach().double().norm()) for (name,p),g in zip(all_params,gradients) if g is not None and bool(torch.isfinite(g).all()) and float(g.detach().double().norm())>0}
    matched=[name for name,out in heads if out.shape==expected.shape[:1]+(c['num_classes'],) and torch.equal(out[sat_start:sat_start+len(y)].detach(),logits.detach())]
    head_names=[n for n in nonzero if any(n.startswith(prefix+'.') for prefix in matched)]
    backbone_names=[n for n in nonzero if not any(n.startswith(prefix+'.') for prefix in matched)]
    assert head_names and backbone_names
    case.update(satellite_head=matched,satellite_head_gradients=head_names,
     satellite_backbone_gradient_tensors=len(backbone_names),satellite_gradient_norm=sum(nonzero.values()))
    loss.register_hook(lambda gradient:case.update(satellite_scalar_gradient=float(gradient)))
   return loss
  if c['family']=='riei':
   full=c['variant']=='full';optimizer=torch.optim.SGD(list(net.ec.parameters())+(list(net.rc.parameters()) if full else list(net.fed.parameters())),lr=c['lr'])
   fed=torch.optim.SGD(net.fed.parameters(),lr=c['lr']) if full else None
  else:optimizer=torch.optim.Adam(net.parameters(),lr=c['lr']);fed=None
  F.cross_entropy=track_ce
  try:
   rec=step(net,c,x,y,rx,optimizer,lambda *args:(satellite,sat_info),.75,bi,
     stage=stage,fed_optimizer=fed,unlabeled=u,native_views=views,author=author,unlabeled_receivers=urx)
  finally:
   F.cross_entropy=original
   for h in handles:h.remove()
  assert any(torch.equal(a,expected) for a in calls)
  assert abs(case['satellite_scalar_gradient']-.68)<1e-6
  expected_loss=rec['identity_ce']+.68*rec['satellite_ce']+rec['native_penalty']+c['pl_weight']*rec['pl_loss']
  assert abs(rec['loss']-expected_loss)<max(2e-4,3e-6*abs(rec['loss']))
  case.update(row_id=c['row_id'],family=c['family'],variant=c['variant'],regime=c['training_regime'],stage=stage,
   status='PASS',batch_size=len(y),joined_examples=len(expected),satellite_offset=sat_start,
   joined_layout_matches=True,selected_checkpoint_telemetry_positive=True,
   satellite_CE=rec['satellite_ce'],satellite_samples=rec['satellite_samples'],applied_samples=rec['satellite_applied_samples'],
   PL_seen=rec['pl_seen'],satellite_scenario=rec['satellite_scenario'],ephemeral_optimizer_steps=rec['optimizer_steps'])
  cases.append(case)
  del net,optimizer,payload,heads,calls,handles,all_params,parts,expected,ce_calls
  gc.collect()
assert len(cases)==20
print(json.dumps(dict(status='PASS',cases=cases,channel_cases=channels,target_reads=0,model_files_written=0,
    device='cpu',threads=2,formal_training_started=0)))
'''
# No target artifacts are read; outer process also checks every frozen config.
outer=r'''
import json,os,subprocess
from pathlib import Path
rt=RUNTIME;items=ITEMS
for item in items:
 c=json.loads((Path(item['root'])/item['row_id']/'source/freeze.json').read_text())['config']
 if any(c.get(k)!=v for k,v in item['config'].items()):raise ValueError('Stored source config changed: '+item['row_id'])
env=dict(os.environ,PYTHONPATH=os.pathsep.join([rt['release_root'],rt['release_root']+'/code',rt['numpy_deps']]),
 CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
p=subprocess.run([rt['python'],'-u','-'],input=INNER,text=True,capture_output=True,cwd=rt['release_root'],env=env,timeout=50)
if p.returncode:raise RuntimeError(p.stderr+'\n'+p.stdout[-1000:])
out=json.loads(p.stdout);out['all_frozen_config_comparisons']=len(items);print(json.dumps(out))
'''.replace('RUNTIME',repr(rt)).replace('ITEMS',repr(items)).replace('INNER',repr(inner.replace('RUNTIME',repr(rt))))
import ast
ast.parse(inner);ast.parse(outer)
result=ssh(outer)
(P/'source_gradient_probes.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(dict(status=result['status'],real_checkpoint_stage_cases=len(result['cases']),
    actual_channel_cases=len(result['channel_cases']),all_frozen_configs=result['all_frozen_config_comparisons'],target_reads=0,N607_GPU_training_processes_started=0)))
