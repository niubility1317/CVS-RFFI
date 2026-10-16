"""Fresh F training, original hidden-U rule, one fixed held-source readout."""
from contextlib import contextmanager
import copy
import gzip
import json
import sys
from pathlib import Path
import torch
from . import design as d,model as api
from experiments.cvs_rdrop_sinc_units.optimizer import native_scope
from experiments.cvs_sigtime_locked20.source import Observer as FoldObserver
from experiments.cvs_feature_sequence50.source import train as shared_train


class RecordingLoader:
    def __init__(self,loader,role,stream):self.loader=loader;self.role=role;self.stream=stream
    def __getattr__(self,key):return getattr(self.loader,key)
    def __len__(self):return len(self.loader)
    def __iter__(self):
        for batch in self.loader:
            x,y,domain,meta=batch
            if 1 in meta['rx_i'].tolist() or (self.role=='U' and bool((y>=0).any())):raise ValueError('Held RX or U truth entered training')
            keys=('eq_i','rx_i','day_i','sig_i','base_index')
            ids=list(zip(*(meta[k].tolist() for k in keys)))
            self.stream.write(json.dumps(dict(role=self.role,ids=ids))+'\n')
            yield batch


class Observer:
    source_contract=FoldObserver.source_contract
    evaluate=FoldObserver.evaluate
    def __init__(self,c):self.c=c;self.optimizer=None
    def data(self,native,ctx,out):
        FoldObserver.data(self,native,ctx,out)
        self.sequence=gzip.open(Path(out)/'training_sequence.jsonl.gz','xt',encoding='utf-8',compresslevel=3)
        for name,role in (('train_loader','L'),('unlabeled_loader','U')):ctx[name]=RecordingLoader(ctx[name],role,self.sequence)
    def initialize(self,m,c,out):
        self.model=m;self.reader=copy.deepcopy(m).eval();self.out=Path(out);(self.out/'source_readouts').mkdir()
        d.write(self.out/'actual_modules.json',{k:str(v.__file__) for k,v in sys.modules.items() if k in ('SSDG.train_ssdg','DataAugmentation','leo_practical.channel','leo_practical.residual')})
    @contextmanager
    def native_scope(self,native):
        original=native.deepcopy
        def copied(value,*args,**kwargs):
            result=original(value,*args,**kwargs)
            if value is getattr(self,'model',None):self.teacher=result
            return result
        native.deepcopy=copied
        try:
            with native_scope(self,'CENTER_UNITS'):yield
        finally:
            native.deepcopy=original
            if hasattr(self,'sequence'):self.sequence.close()
    def epoch(self,m,ctx,rows,out,device):
        steps=len(rows)*222
        if m._feature_l_steps!=steps or m._timing_calls!=steps or m._timing_L_packets!=steps*768 or self.optimizer._sinc_physical_clip_calls!=steps or self.optimizer._sinc_unit_steps!=steps:
            raise ValueError('Actual F/timing/default-head budget differs')
        d.write(out/'timing_progress.json',dict(epoch=len(rows),steps=steps,L_packets=m._timing_L_packets,source_only=True,head_fit_updates=0))
    def finish(self):
        done=d.read(self.out/'completion.json');payload=torch.load(self.out/'final_ssdg.pth',map_location='cpu',weights_only=False)
        if done['epoch']!=20 or done['optimizer_steps']!=4440 or done['initialization']!='scratch' or done['checkpoint_sources'] or payload['args']['baseline_ckpt'] or payload['args']['teacher_ckpt'] or payload['args']['use_aug'] or payload['args'].get('source_fold_held_rx')!=1 or payload['checkpoint_selection']!='final_only' or payload['a1_ema_successful_updates']!=4440 or payload.get('ema_model') is None:
            raise ValueError('Complete own F scratch20/default-head/EMA source model required')
        initial=torch.load(self.out/'initial_smoke.pt',map_location='cpu',weights_only=False)['model']
        if any(not torch.equal(v,self.model.state_dict()[n].detach().cpu()) for n,v in payload['model'].items()):raise ValueError('Frozen state changed')
        W='id_backbone.encoder.core.id_backbone.cls_head.weight'
        if payload['model'][W].shape!=(6,160) or torch.equal(initial[W],payload['model'][W]):raise ValueError('Default head must update jointly')
        if self.optimizer._sinc_physical_clip_calls!=4440 or len(self.model._timing_details)!=4440:raise ValueError('Missing actual steps')
        held=self.evaluate(payload['model']);d.write(self.out/'source_readouts/FINAL.json',held)
        f1=d.macro_f1(held['labels'],held['predictions'])
        with gzip.open(self.out/'timing_steps.jsonl.gz','xt',encoding='utf8') as f:
            for record in self.model._timing_details:f.write(json.dumps(record,allow_nan=False)+'\n')
        with gzip.open(self.out/'unit_steps.jsonl.gz','xt',encoding='utf8') as f:
            for record in self.optimizer._sinc_unit_lr_records:f.write(json.dumps(record,allow_nan=False)+'\n')
        compact={k:v for k,v in held.items() if k not in ('physical_ids','labels','predictions')}
        student=self.model.id_backbone.encoder;teacher=self.teacher.id_backbone.encoder
        result=dict(status='SOURCE_TIMING_COMPLETE',arm=self.c['arm'],actual_steps=4440,final_L_CE=done['final_training_L_CE'],held_default_head=compact,macro_F1=f1,seen_V=done['final_source_metrics'],L_forward_packets=self.model._timing_L_packets,student_train_forward_packets=student._timing_train_packets,student_eval_forward_packets=student._timing_eval_packets,teacher_train_forward_packets=teacher._timing_train_packets,teacher_eval_forward_packets=teacher._timing_eval_packets,held_forward_packets=1260,unused_fit_anchor_reads=144,smoke_forward_packets=2,elapsed_seconds=done['elapsed_seconds'],default_head_joint=True,head_fit_updates=0,target_access=False,U_truth_read=False,checkpoint_sources=[],ordinary_L=False,Ag=self.c['training_mechanism']['Ag'],As=False,model='F/X3/unifiedSinc/joint160')
        d.write(self.out/'timing_result.json',result)
        return result


def train(c):
    observer=Observer(c);shared_train(c,design=d,model_api=api,observer=observer)
    return observer.finish()
