"""Materialize lawful source U IQ with TX hidden at the dataset boundary."""
import time
import torch
from . import design as d
from experiments.cvs_stage2_hybrid.semisupervised import validate_unlabeled


def build():
 from experiments.cvs_phase1_stack.runtime import installed
 from experiments.cvs_stage2_decouple import design as parent
 from experiments.cvs_stage2_decouple.data import _preprocessing
 from cvsrffi.xuc_fusion.native import role_ids_from_native
 from cvsrffi.game_tracking.data import opaque_id
 expected=d.read(d.SOURCE);started=time.perf_counter();pc=d.parent_config(d.SEEDS[0])
 with installed(pc) as native:
  args=parent.make_args(pc,'cpu')
  if not args.a1_source_screen_only or args.use_concat_sat_channel_aug:raise ValueError('Source-only cache required')
  context=native._build_ssdg_wisig_data(args,torch.device('cpu'))
  if context['named_test_loaders'] or role_ids_from_native(context)!=expected['role_ids']:raise ValueError('Source roles differ or target loader constructed')
  dataset=context['unlabeled_loader'].dataset
  if not getattr(dataset,'hide',False):raise ValueError('U dataset must hide TX truth before method access')
  preprocessing=_preprocessing(dataset)
  ids=[opaque_id(item) for item in dataset.index]
  if len(ids)!=56700 or sorted(ids)!=sorted(expected['role_ids']['U_s']):raise ValueError('Exact U_s role required')
  if set(ids)&(set(expected['role_ids']['L_s'])|set(expected['role_ids']['V'])):raise ValueError('Overlapping source roles')
  rows=[];metadata={k:[] for k in ('rx','day','domain','eq','sig','base_index')}
  for i in range(len(dataset)):
   x,y,domain,meta=dataset[i][:4]
   if int(y)!=-1 or meta.get('tx_label_visible') is not False or any(k in meta for k in ('tx','tx_i','true_tx_i','tx_label','y_tx')):raise ValueError('U truth leaked')
   rows.append(torch.as_tensor(x).detach().cpu().float().contiguous())
   metadata['domain'].append(int(domain))
   for key in ('rx','day','eq','sig','base_index'):metadata[key].append(int(meta[key+'_i' if key!='base_index' else key]))
  cache=dict(x=torch.stack(rows),ids=ids,role='U_s',preprocessing=preprocessing,**{k:torch.tensor(v,dtype=torch.long) for k,v in metadata.items()})
  validate_unlabeled(cache,expected['role_ids']['U_s']);del context
 folder=d.UNLABELED_CACHE.parent;folder.mkdir(exist_ok=False)
 torch.save(cache,d.UNLABELED_CACHE)
 result=dict(status='SOURCE_UNLABELED_READY',role='U_s',count=56700,source_contract=expected,TX_truth_exposed=False,
  target_loaders_constructed=False,identity_updates=0,V_fit=False,elapsed_seconds=time.perf_counter()-started,cache=d.UNLABELED_CACHE.as_posix())
 d.write(folder/'complete.json',result)
 return result
