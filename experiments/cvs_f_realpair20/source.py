from collections import Counter
from contextlib import contextmanager
import gzip,json,time
from pathlib import Path
from types import SimpleNamespace
import torch
from . import design as d,model as api
from .pairing import Bank,FIT
from experiments.cvs_ordinary_fold20.source import Observer as BasicObserver,shared_train
from experiments.cvs_ordinary_fold20.streams import stream
from experiments.cvs_raw_anchor20.source import Observer as FObserver,F_optimizer_scope
from experiments.cvs_f_crossrx_practical20.source import evaluate_practical

class NativeView:
    def __init__(self,inner,observer):object.__setattr__(self,'_inner',inner);object.__setattr__(self,'_observer',observer)
    def __getattr__(self,k):return getattr(self._inner,k)
    def __setattr__(self,k,v):setattr(self._inner,k,v)
    def __call__(self,x,*args,**kw):
        o=self._observer;m=o.m
        if m._pair_primary is not None or m._pair_secondary is not None or m._pair_secondary_domains is not None or o._current_L_meta is None:raise ValueError('Previous real L pair not consumed')
        meta,labels=o._current_L_meta,o._current_L_labels;o._current_L_meta=o._current_L_labels=None
        if kw.get('labels') is None or kw['labels'].detach().cpu().tolist()!=labels:raise ValueError('Native true L labels must match physical pairing labels')
        primary=self._inner(x,*args,**kw);step=m._anchor_calls+1;started=time.perf_counter()
        other,domains,records=o.bank.select(labels,meta,step,o.c['arm'],x)
        # The same original augmentor/schedule is reused, under an independent
        # stream which cannot advance the primary/loader/dropout/U streams.
        with stream(d.SECOND_AUG_SEED,step,x.device):secondary=self._inner._inner(other.to(x.device),*args,**kw)
        if secondary.shape!=x.shape or not bool(torch.isfinite(secondary).all()):raise ValueError('Finite real partner ordinary IQ required')
        o.partner_mse+=float((secondary-other.to(x.device)).square().sum());o.partner_max=max(o.partner_max,float(secondary.detach().abs().max()))
        o.partner_log.write(json.dumps(dict(step=step,records=records),allow_nan=False)+'\n')
        m._pair_primary=primary.detach().clone();m._pair_secondary=secondary.detach().clone();m._pair_secondary_domains=domains.to(x.device)
        o.partner_view_seconds+=time.perf_counter()-started
        return primary

class Observer(FObserver):
    def __init__(self,c):
        super().__init__(c);self._current_L_meta=None;self._current_L_labels=None;self.bank=None;self.partner_mse=0.;self.partner_max=0.;self.partner_view_seconds=0.
    def data(self,native,ctx,out):
        super().data(native,ctx,out);self._bank_dataset=ctx['train_loader'].dataset;self._bank_domain_map=dict(ctx['domain_label_map'])
    def initialize(self,m,c,out):
        super().initialize(m,c,out)
        # Native own-scratch checkpoint no-query smoke has already passed.
        # Fetch only this run's filtered visible L dataset, never U or V.
        started=time.perf_counter()
        with stream(d.PAIR_SEED,0):self.bank=Bank.from_dataset(self._bank_dataset,self._bank_domain_map)
        self.bank_read_seconds=time.perf_counter()-started
        del self._bank_dataset
        self.partner_log=gzip.open(self.out/'real_pair_sequence.jsonl.gz','xt',encoding='utf-8',compresslevel=3)
        d.write(self.out/'real_pair_bank.json',dict(status='VERIFIED',bank=self.bank.summary(),read_after_own_scratch_checkpoint_smoke=True,
            source='Exact current filtered train_loader.dataset',bank_read_seconds=self.bank_read_seconds,
            raw_domain_remap='Original data_ctx domain_label_map, identical to native domain_from_extra before forward',
            new_dataset_roles=False,extra_IQ_transform=False))
    def record(self,role,batch):
        super().record(role,batch)
        if role=='L_s':
            if self._current_L_meta is not None:raise ValueError('Previous L metadata unconsumed')
            self._current_L_meta=batch[3];self._current_L_labels=batch[1].tolist()
    @contextmanager
    def native_scope(self,native):
        with BasicObserver.native_scope(self,native),F_optimizer_scope(self,'CENTER_UNITS'):
            factory=native.make_augmentor;native.make_augmentor=lambda cfg:NativeView(factory(cfg),self)
            try:yield
            finally:
                native.make_augmentor=factory
                if hasattr(self,'partner_log'):self.partner_log.close()
    def epoch(self,m,ctx,rows,out,device):
        if self._current_L_meta is not None or m._pair_primary is not None or m._pair_secondary is not None or m._pair_secondary_domains is not None:raise ValueError('Real partner cache escaped update')
        super().epoch(m,ctx,rows,out,device);epoch=len(rows);steps=epoch*222
        if m._supcon_calls!=steps or m._supcon_applied!=max(0,epoch-5)*222 or m._supcon_pending is not None or m._supcon_handles:raise ValueError('Unchanged original primary-pair SupCon required')
        if self.bank.calls!=steps or self.bank.packets!=steps*128:raise ValueError('One real sameTX L partner per anchor required')
        self.partner_log.flush();self.epochs[-1]['SupCon']=dict(calls=m._supcon_calls,applied_calls=m._supcon_applied,last=m._supcon_records[-1],extra_model_forwards=0)
        self.epochs[-1]['real_pair']=self.bank.summary()
        self.epochs[-1]['partner_ordinary']=dict(calls=steps,packets=steps*128,waveform_MSE=self.partner_mse/(steps*128*512),max_abs=self.partner_max,
            bank_read_seconds=self.bank_read_seconds,partner_view_construction_seconds=self.partner_view_seconds,
            timing_scope='Paired sampler+native ordinary+device transfer+waveform statistics+log/cache, no backbone forward; synchronized by scalar waveform statistics')
        d.write(self.out/'mechanism_progress.json',dict(records=self.epochs,head_fit_updates=0,U_truth_read=False,target_access=False))
    def finish(self):
        result=super().finish();m=self.m
        if self.bank.calls!=4440 or self.bank.packets!=568320 or self.bank.unique!=set(self.bank.index):raise ValueError('Complete E20/all5040 real partners required')
        expected={(t,r,other,day) for t in range(6) for r in FIT for other in FIT for day in (1,2,3) if (r==other)==(self.c['arm']=='SAME_RX')}
        if set(self.bank.coverage)!=expected:raise ValueError('Every trueTX x primaryRX x allowed partnerRX x day cell must be exposed')
        if m._supcon_calls!=4440 or m._supcon_applied!=3330:raise ValueError('Full E20 contrastive schedule required')
        with gzip.open(self.out/'supcon_steps.jsonl.gz','xt',encoding='utf-8') as f:
            for r in m._supcon_records:f.write(json.dumps(r,allow_nan=False)+'\n')
        result['real_pair']=self.bank.summary();result['partner_ordinary']=self.epochs[-1]['partner_ordinary']
        result['supcon']=dict(calls=m._supcon_calls,applied_calls=m._supcon_applied,only_primary_pair=True,extra_model_forwards=0,new_parameters=0,U_truth_read=False)
        started=time.perf_counter();result['practical']=evaluate_practical(self);result['source_practical_seconds']=time.perf_counter()-started;result['source_practical_forward_packets']=6*5400
        d.write(self.out/'real_pair_proof.json',dict(status='VERIFIED',bank=result['real_pair'],partner_ordinary=result['partner_ordinary'],SupCon=result['supcon'],
            pair_weights=[.5,.5],L_forwards_per_sample=4,all_L_views_supervised=True,same_IQ_within_each_RDrop_pair=True,
            partner_has_own_true_RX_day_domains=True,synthetic_hardware=False,IQ_mixing=False,new_feature_alignment=False,head_fit_updates=0,target_access=False))
        d.write(self.out/'result.json',result);return result

def train(c):
    observer=Observer(c)
    view=SimpleNamespace(**{k:getattr(d,k) for k in ('validate','make_args','read','write','SOURCE','FULL_FP32_POLICY','ROOT','UPDATES')},
        source_validation=lambda model,loader,device:d.prior.old.old.source_validation(model,loader,device,fit=observer.fit))
    shared_train(c,design=view,model_api=api,observer=observer)
    return observer.finish()
