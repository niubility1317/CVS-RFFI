"""Reuse validated IQ roles while retaining native temporal-gate metadata."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from tools.local_phase1_data import SourceRoleDataset
from .baseline import original_config,make_args
ROLES={'L_s':'train_loader','U_s':'unlabeled_loader','V':'val_loader'}
SPLITS={'L_s':'ssdg_labeled_tx_visible','U_s':'ssdg_unlabeled_tx_hidden','V':'ssdg_source_v_select'}

def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,v):
 with Path(p).open('x',encoding='utf-8') as f:json.dump(v,f,indent=2,allow_nan=False)

def prepare(profile_path,output):
 profile=read(profile_path);cache=Path(profile['dataset']['source_cache']);manifest=read(cache/'manifest.json');out=Path(output)
 if manifest['native_role_ids']!='EXACT_MATCH' or manifest['U_TX_labels_exported'] or manifest['status']!='READY':raise ValueError('Use the existing source-only native cache')
 if out.exists():raise FileExistsError('Preserve existing native metadata cache')
 from experiments.cvs_phase1_stack.runtime import installed
 c=original_config(2026092701);a=make_args(2026092701,'cpu');a.wisig_pkl=profile['dataset']['manysig_pkl'];a.num_workers=0;torch.set_num_threads(2)
 if not a.a1_source_screen_only:raise ValueError('Metadata materialization cannot construct target loaders')
 with installed(c) as native:
  from cvsrffi.game_tracking.data import opaque_id
  ctx=native._build_ssdg_wisig_data(a,torch.device('cpu'))
  if ctx['named_test_loaders'] or ctx['num_domains']!=15:raise ValueError('Original source-only context differs')
  out.mkdir(parents=True);counts={};alignment=[]
  for role,key in ROLES.items():
   original=ctx[key].dataset;cached=SourceRoleDataset(cache,role)
   if len(original)!=len(cached):raise ValueError('Role count differs')
   positions={str(value):i for i,value in enumerate(cached.ids)}
   if len(positions)!=len(cached):raise ValueError('Existing source IDs must be unique')
   # Join NEW non-TX metadata to existing opaque physical IDs; no IQ rewriting,
   # dataset rehash, new capsule validation or U TX export.
   values=np.full((len(cached),3),-1,dtype=np.int64)
   for ordinal,item in enumerate(original.index):
    index=positions.pop(opaque_id(item));values[index]=[int(item.eq_i),int(item.sig_i),int(original.selected[ordinal])]
    if item.rx_i!=int(cached.rx[index]) or item.day_i!=int(cached.day[index]):raise ValueError('Metadata/cache alignment differs')
   if positions or (values<0).any():raise ValueError('Metadata ID join incomplete')
   np.save(out/(role+'.npy'),values,allow_pickle=False);counts[role]=len(cached)
   for ordinal in sorted({0,len(cached)//2,len(cached)-1}):
    item=original.index[ordinal];index={str(v):i for i,v in enumerate(cached.ids)}[opaque_id(item)]
    ox,oy,od,om=original[ordinal];cx,cy,cd,cm=cached[index]
    if not torch.equal(ox,cx) or int(od)!=cd or int(om['base_index'])!=values[index,2] or (role=='U_s' and (oy!=-1 or cy!=-1)):raise ValueError('Native IQ/metadata alignment smoke failed')
   alignment.append(dict(role=role,native_packet_smoke=3,IQ_exact=True,domain_exact=True,TX_hidden=role=='U_s'))
  context={k:v for k,v in ctx.items() if k not in ('train_loader','probe_train_loader','unlabeled_loader','source_calibration_loader','val_loader','balanced_train_sampler','named_test_loaders')}
  context['domain_label_map']=[[k,v] for k,v in ctx['domain_label_map'].items()]
  write(out/'context.json',context);write(out/'manifest.json',dict(status='READY',schema='source_native_gate_metadata_v1',roles=counts,fields=['eq_i','sig_i','base_index'],U_TX_labels_exported=False,target_constructed=False,source_IQ='Existing validated source cache unchanged',alignment=alignment,source_role_ids='Reuse EXACT_MATCH cache physical IDs; new metadata joined to those IDs'))
 print(json.dumps(dict(status='READY',roles=counts,U_TX_labels_exported=False,target_reads=0,optimizer_updates=0)),flush=True)

class NativeCachedRole(SourceRoleDataset):
 def __init__(self,cache,metadata,role,split_source=None):
  super().__init__(cache,role);self.samples_read=0;self.gate=np.load(Path(metadata)/(role+'.npy'),mmap_mode='r',allow_pickle=False);self.split_source=split_source or SPLITS[role]
  if self.gate.shape!=(len(self),3) or self.gate.dtype!=np.int64:raise ValueError('Native temporal metadata shape differs')
 def __getitem__(self,index):
  self.samples_read+=1;x,y,domain,meta=super().__getitem__(index);eq,sig,base=self.gate[index]
  meta.update(eq_i=int(eq),sig_i=int(sig),base_index=int(base),split_source=self.split_source)
  if self.role!='U_s':meta['tx_i']=int(y)
  return x,y,domain,meta

def build_context(native,args,device,profile,metadata):
 cache=Path(profile['dataset']['source_cache']);m=read(cache/'manifest.json');proof=read(Path(metadata)/'manifest.json')
 if m['status']!='READY' or m['native_role_ids']!='EXACT_MATCH' or m['U_TX_labels_exported'] or proof['status']!='READY' or proof['U_TX_labels_exported'] or proof['target_constructed']:raise ValueError('Existing source handles/hidden U differ')
 if args.use_concat_sat_channel_aug or not args.a1_source_screen_only or args.use_tx_rx_balanced_sampler or args.num_workers!=0:raise ValueError('Original source-only random-L/no-satellite recipe required')
 saved=read(Path(metadata)/'context.json');domains={int(k):int(v) for k,v in saved['domain_label_map']}
 L=NativeCachedRole(cache,metadata,'L_s');U=NativeCachedRole(cache,metadata,'U_s');V=NativeCachedRole(cache,metadata,'V');cal=NativeCachedRole(cache,metadata,'V','ssdg_source_v_cal')
 if (len(L),len(U),len(V))!=(6300,56700,27000):raise ValueError('Original7/63/30 source roles required')
 loader=lambda ds,batch,shuffle,drop:native.make_loader(ds,batch,shuffle,0,device,drop,int(args.prefetch_factor))
 ctx=dict(saved,train_loader=loader(L,128,True,True),probe_train_loader=loader(L,args.eval_batch_size,False,False),unlabeled_loader=loader(U,256,bool(getattr(args,'u_unlabeled_shuffle',True)),False),source_calibration_loader=loader(cal,args.eval_batch_size,False,False),val_loader=loader(V,args.eval_batch_size,False,False),balanced_train_sampler=None,named_test_loaders={},domain_label_map=domains)
 if len(ctx['train_loader'])!=49 or len(ctx['unlabeled_loader'])!=222:raise ValueError('Native random-L/U256 step budget differs')
 # The native split receipt remains native telemetry, not a new permission gate.
 # Update its model seed while retaining the SAME source physical partition.
 ctx['split_info']=dict(saved['split_info']);receipt=dict(ctx['split_info']['source_split_receipt']);receipt['seed']=int(args.seed);receipt.pop('split_manifest_sha256',None);receipt['split_manifest_sha256']=native._canonical_json_sha256(receipt);ctx['split_info']['source_split_receipt']=receipt
 args.num_classes=6
 return ctx

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--profile',required=True);p.add_argument('--output',required=True);a=p.parse_args();prepare(a.profile,a.output)
