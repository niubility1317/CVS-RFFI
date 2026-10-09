"""Synthetic production-loop integration, including late pseudo-label phase."""
import argparse
from contextlib import redirect_stdout
from copy import deepcopy
import json
from pathlib import Path
import torch
from torch.utils.data import DataLoader,Dataset
from experiments.cvs_multi_state_action import design as d
from experiments.cvs_multi_state_action.runtime import installed
from experiments.cvs_multi_state_action.source import incremental_writer
from experiments.cvs_phase1_stack.fast_execution import configure
from experiments.cvs_phase1_stack.runtime import blind_sample
from experiments.cvs_equivariant_identity.precision import numerical_context

class Synthetic(Dataset):
 def __init__(self,role):
  self.role=role;gen=torch.Generator().manual_seed(819)
  self.x=torch.randn(384,2,256,generator=gen);self.offset={'L':0,'U':10000,'V':20000}[role]
 def __len__(self):return len(self.x)
 def __getitem__(self,i):
  y=i%6;rx=1 if (i//6)%2 else 3
  v=(self.x[i],y,int(rx==3),dict(rx_i=rx,day_i=1,eq_i=1,sig_i=i+self.offset,base_index=i+self.offset,tx_i=y))
  return blind_sample(v) if self.role=='U' else v

class Finished(Exception):pass

def run(output,device='cpu',arms=None):
 output=Path(output);output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2);results=[]
 for arm in (arms or d.ARMS):
  c=d.config(next(r for r in d.rows() if r['arm']==arm));a=configure(d.make_args(c,device));a.output_dir=str(output/arm)
  a.batch_size=32;a.muse_unlabeled_batch_size=32;a.eval_batch_size=32
  a.source_val_heavy_eval_start_epoch=9999;a.rc4_calibration_update_epochs='1';a.daot_diagnostic_epochs='1';a.rc4_gradient_telemetry_epochs='1'
  seen=[];snapshot={}
  with numerical_context(d.FULL_FP32_POLICY),installed(dict(c,output_root=a.output_dir)) as n,incremental_writer(n):
   originals={k:getattr(n,k) for k in ('_build_ssdg_wisig_data','build_baseline_model','_write_ssdg_epoch_telemetry','_should_run_source_val_heavy_eval')}
   def data(*args):
    l=DataLoader(Synthetic('L'),batch_size=32,shuffle=True);u=DataLoader(Synthetic('U'),batch_size=32);v=DataLoader(Synthetic('V'),batch_size=32)
    return dict(train_loader=l,probe_train_loader=l,unlabeled_loader=u,val_loader=v,source_calibration_loader=v,
     named_test_loaders={},domain_label_map={1:0,3:1},num_domains=2,input_len=256,class_id_to_tx=d.CLASSES,
     balanced_train_sampler=None,split_info=dict(labeled_size=384,unlabeled_size=384,source_val_size=384,
      source_calibration_size=384,source_selection_size=384,mode='synthetic',named_test_meta={},source_split_receipt={}))
   def build(*args):
    m=originals['build_baseline_model'](*args);snapshot['initial']={k:v.detach().cpu().clone() for k,v in m.state_dict().items()};return m
   def telemetry(cp,jp,rows):
    originals['_write_ssdg_epoch_telemetry'](cp,jp,rows);seen.append(int(rows[-1]['epoch']))
    if seen[-1]==200:raise Finished()
   n._build_ssdg_wisig_data=data;n.build_baseline_model=build;n._write_ssdg_epoch_telemetry=telemetry
   n._should_run_source_val_heavy_eval=lambda *a:False
   n.range=lambda *a:iter((1,21,131,200)) if a==(1,201) else range(*a)
   try:
    with (output/(arm+'.log')).open('x',encoding='utf-8') as log,redirect_stdout(log):
     try:n.train(a)
     except Finished:pass
    if seen!=[1,21,131,200]:raise ValueError('Synthetic production epochs incomplete')
    aux=torch.load(Path(a.output_dir)/'auxiliary_final.pth',map_location='cpu',weights_only=False)
    if aux['target_access'] or aux['inference_uses_auxiliary']:raise ValueError('Inference auxiliary leak')
    if arm in ('unified','LTR','LTR_EG'):
     ex=aux['mechanism_execution']
     if ex['fit_steps'].get('receiver',0)<=0 or ex['receiver_relations_total']<=0:raise ValueError('R never executed in production loop')
    results.append(dict(arm=arm,status='PASS',epochs=seen,execution=aux['mechanism_execution']))
   finally:
    del n.range
    for k,v in originals.items():setattr(n,k,v)
 d.write(output/'completion.json',dict(status='PASS',device=device,rows=results,synthetic=True,
  target_access=False,full_budget_verified=False,production_native_hook=True,production_writer=True))
 return results

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--device',default='cpu');p.add_argument('--arms',nargs='+');a=p.parse_args()
 print(json.dumps(run(a.output,a.device,a.arms)))
