"""Synthetic native L/U/ownEMA/physical-Sinc backend check, no dataset IQ."""
import argparse,json,os
from pathlib import Path
from types import SimpleNamespace
import torch
from . import baseline,model as api,design as d
from experiments.cvs_phase1_stack.runtime import installed
from experiments.cvs_rdrop_sinc_units.optimizer import native_scope
from experiments.cvs_feature_sequence50.source import native_default_head_only
from experiments.cvs_equivariant_identity.precision import numerical_context

class Toy(torch.utils.data.Dataset):
 def __init__(self,n,role):self.n=n;self.role=role
 def __len__(self):return self.n
 def __getitem__(self,index):
  x=torch.randn(2,256,generator=torch.Generator().manual_seed(index+991))
  y=-1 if self.role=='U' else index%6
  meta=dict(rx_i=1,day_i=1,eq_i=0,sig_i=index,base_index=index,split_source='ssdg_unlabeled_tx_hidden' if self.role=='U' else 'ssdg_labeled_tx_visible',tx_label_visible=self.role!='U')
  return x,y,5,meta

def run(profile,metadata,out):
 torch.set_num_threads(2);out=Path(out);out.mkdir(parents=True,exist_ok=False);os.environ['TORCHINDUCTOR_CACHE_DIR']=str(out/'torch_cache')
 c=d.config(d.rows()[0],str(out/d.RUN),profile,metadata);a=d.make_args(c,'cpu')
 # Test-only2epochs/10updates; no source IQ, no candidate weights are inherited.
 a.epochs=2;a.label_epochs=1;a.pseudo_epochs=1;a.output_dir=str(out/'native');a.device='cpu';a.test_eval_start_epoch=999999;a.run_id='SYNTHETIC_NATIVE_BACKEND_SMOKE';a.candidate_id='NO_SOURCE_NO_TARGET'
 owner=SimpleNamespace(model=None,optimizer=None);events=[]
 with numerical_context(d.FULL_FP32_POLICY),installed(c) as n,native_default_head_only(n) as guard:
  old_data=n._build_ssdg_wisig_data;old_model=n.build_baseline_model
  def data(args,device):
   loader=lambda ds,batch,shuffle,drop:n.make_loader(ds,batch,shuffle,0,device,drop,2)
   split=json.loads((Path(metadata)/'context.json').read_text())['split_info'];split.update(labeled_size=140,unlabeled_size=1260,source_val_size=600,source_calibration_size=600,source_selection_size=600);split['source_split_receipt']=n._build_source_split_receipt(seed=a.seed,split_mode=a.split_mode,source_days=[1,2,3],target_days=[0],source_receivers=[1,3,4,6,8],target_receivers=[0],labeled_indices=list(range(140)),unlabeled_indices=list(range(140,1400)),source_validation_indices=list(range(1400,2000)),wisig_pkl_sha256='',requested_labeled_ratio=.07,requested_unlabeled_ratio=.63,requested_source_val_ratio=.3)
   return dict(train_loader=loader(Toy(140,'L'),128,True,True),unlabeled_loader=loader(Toy(1260,'U'),256,True,False),val_loader=loader(Toy(600,'V'),256,False,False),probe_train_loader=loader(Toy(140,'L'),256,False,False),source_calibration_loader=loader(Toy(600,'V'),256,False,False),balanced_train_sampler=None,named_test_loaders={},domain_label_map={5:0},num_domains=15,input_len=256,class_id_to_tx=list(d.CLASSES),split_info=split)
  def build(args,device):
   from experiments.cvs_source_identity_readout.input_policy import apply
   m=old_model(args,device);m._identity_input_policy=apply(m);m=api.install(m,c);owner.model=m;return m
  n._build_ssdg_wisig_data=data;n.build_baseline_model=build
  try:
   with native_scope(owner,'CENTER_UNITS'):code=n.train(a)
  finally:n._build_ssdg_wisig_data=old_data;n.build_baseline_model=old_model
 if code:raise ValueError('Synthetic native backend failed')
 cp=torch.load(out/'native/final_ssdg.pth',weights_only=False,map_location='cpu');steps=owner.optimizer._sinc_unit_steps
 if cp['epoch']!=2 or cp['a1_ema_successful_updates']!=steps or steps!=10 or not cp.get('ema_model') or owner.model._legacy_calls!=10 or owner.model._supcon_calls!=10 or owner.optimizer._sinc_physical_clip_calls!=10 or guard['disabled_calls']!=1:raise ValueError('Native U/ownEMA/physicalSinc backend budget differs')
 value=dict(status='PASS',purpose='Synthetic native backend compatibility only',synthetic_epochs=2,updates=10,L_stage=True,U_stage=True,ownEMA_updates=10,physical_Sinc_clips=10,RDrop_calls=10,U_TX_truth_hidden=True,source_IQ_reads=0,target_reads=0,head_fit_updates=0,formal_candidate_budget_unchanged='50=33L+17U,11100updates',inherited_by_candidate=False)
 (out/'result.json').write_text(json.dumps(value,indent=2),encoding='utf-8');print(json.dumps(value),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--profile',required=True);p.add_argument('--metadata',required=True);p.add_argument('--output',required=True);a=p.parse_args();run(a.profile,a.metadata,a.output)
