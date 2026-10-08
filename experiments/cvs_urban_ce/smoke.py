"""Synthetic native-loop integration at the actual late-mechanism epochs."""
import argparse
import contextlib
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import torch
from torch.utils.data import DataLoader,Dataset
from experiments.cvs_urban_ce.design import *
sys.path[:0]=[str(ROOT),str(ROOT/'code')]
from experiments.cvs_urban_ce.runtime import installed
from experiments.cvs_equivariant_identity.precision import numerical_context


class Synthetic(Dataset):
    def __init__(self,hidden=False,muse=False):self.hidden=hidden;self.muse=muse
    def __len__(self):return 24
    def __getitem__(self,i):
        g=torch.Generator().manual_seed(700+i);x=torch.randn(2,256,generator=g)
        y=i%6;d=(i//6)%2
        meta=dict(rx_i=d,day_i=1,eq_i=1,sig_i=i,base_index=i)
        if self.muse:return x,d,dict(meta,tx_label_visible=False)
        return x,-1 if self.hidden else y,d,dict(meta,**({} if self.hidden else {'tx_i':y}))


class Finished(Exception):pass


def smoke(output,device='cpu'):
    torch.set_num_threads(2);output=Path(output);output.mkdir(parents=True,exist_ok=False)
    findings=[]
    for arm in ARMS:
        stage='urban_ce'
        row=next(r for r in rows() if r['stage']==stage and r['arm']==arm)
        c=config(row)
        a=make_args(c,device);a.output_dir=str(output/(stage+'-'+arm));a.batch_size=24;a.eval_batch_size=24
        a.source_val_heavy_eval_start_epoch=9999;a.rc4_calibration_update_epochs='1'
        a.daot_diagnostic_epochs='1';a.rc4_gradient_telemetry_epochs='1'
        seen=[]
        with numerical_context(FULL_FP32_POLICY),installed(c) as n:
            originals={k:getattr(n,k) for k in ['build_baseline_model','_build_ssdg_wisig_data','_write_ssdg_epoch_telemetry','_should_run_source_val_heavy_eval']}
            def build(args, dev):
                m=originals['build_baseline_model'](args,dev)
                path=output/(arm+'-own-scratch.pt');torch.save(m.state_dict(),path)
                m.load_state_dict(torch.load(path,map_location=dev,weights_only=False),strict=True)
                m.eval()
                with torch.no_grad(): z=m(torch.zeros(2,2,256,device=dev))
                assert z.shape==(2,6) and torch.isfinite(z).all()
                return m
            n.build_baseline_model=build
            def ctx(*args):
                l=DataLoader(Synthetic(),batch_size=24);u=DataLoader(Synthetic(True,a.use_muse_ssdg),batch_size=24)
                return dict(train_loader=l,probe_train_loader=l,unlabeled_loader=u,val_loader=l,source_calibration_loader=l,
                    named_test_loaders={},domain_label_map={0:0,1:1},num_domains=2,input_len=256,class_id_to_tx=CLASSES,
                    balanced_train_sampler=None,split_info=dict(labeled_size=24,unlabeled_size=24,source_val_size=24,
                        source_calibration_size=24,source_selection_size=24,mode='synthetic',named_test_meta={},source_split_receipt={}))
            def telemetry(cp,jp,records):
                originals['_write_ssdg_epoch_telemetry'](cp,jp,records)
                seen.append(records[-1]['epoch'])
                native_lr=records[-1]['lr']
                wanted=cosine_lr(a.lr,seen[-1]) if c['lr_schedule']=='cosine' else a.lr
                if abs(native_lr-wanted)>1e-10: raise ValueError(('Actual LR mismatch',native_lr,wanted))
                if seen[-1]==200:raise Finished()
            n._build_ssdg_wisig_data=ctx;n._write_ssdg_epoch_telemetry=telemetry
            n._should_run_source_val_heavy_eval=lambda *args:False
            n.range=lambda *args:iter([1,80,131,132,200]) if args==(1,201) else range(*args)
            try:
                with (output/(stage+'-'+arm+'.log')).open('x',encoding='utf-8') as log,contextlib.redirect_stdout(log):
                    try:n.train(a)
                    except Finished:pass
                assert seen==[1,80,131,132,200],seen
                findings.append(dict(stage=stage,arm=arm,epochs=seen,source='synthetic',status='PASS'))
            finally:
                del n.range
                for k,v in originals.items():setattr(n,k,v)
    write(output/'completion.json',dict(status='PASS',device=device,rows=findings,target_access=False,formal_weights=False))
    print(json.dumps(findings))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--device',default='cpu');a=p.parse_args();smoke(a.output,a.device)
