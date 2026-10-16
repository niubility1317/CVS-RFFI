from collections import Counter
from contextlib import contextmanager
import gzip,json,time
from types import SimpleNamespace
from . import design as d,model as api
from .attenuation import LocalAttenuation
from experiments.cvs_f_crossrx_practical20.source import evaluate_practical
from experiments.cvs_ordinary_fold20.source import Observer as BasicObserver,shared_train
from experiments.cvs_raw_anchor20.source import Observer as FObserver,F_optimizer_scope

class NativeView:
    def __init__(self,inner,observer):object.__setattr__(self,'_inner',inner);object.__setattr__(self,'_observer',observer)
    def __getattr__(self,k):return getattr(self._inner,k)
    def __setattr__(self,k,v):setattr(self._inner,k,v)
    def __call__(self,x,*args,**kw):
        o=self._observer;m=o.m
        if m._pair_primary is not None or m._pair_secondary is not None or o._current_L_meta is None:
            raise ValueError('One fresh physical L view pair required')
        meta,labels=o._current_L_meta,o._current_L_labels;o._current_L_meta=o._current_L_labels=None
        if kw.get('labels') is None or kw['labels'].detach().cpu().tolist()!=labels:raise ValueError('Native true L labels must match physical labels')
        primary=self._inner(x,*args,**kw);started=time.perf_counter();step=m._anchor_calls+1
        secondary,ids,stats=o.local.apply(primary,meta,step)
        for tx,p,start in zip(labels,ids,stats['starts']):o.local_coverage[(tx,p[1],p[2],start//16)]+=1
        o.local_trace.write(json.dumps(dict(step=step,ids=ids,**stats),allow_nan=False)+'\n')
        o.local_seconds+=time.perf_counter()-started
        m._pair_primary=primary.detach().clone();m._pair_secondary=secondary.detach().clone()
        return primary

class Observer(FObserver):
    def __init__(self,c):
        super().__init__(c);self._current_L_meta=None;self._current_L_labels=None
        self.local=LocalAttenuation(d.LOCAL_SEED,self.fit);self.local_coverage=Counter();self.local_seconds=0.
    def record(self,role,batch):
        super().record(role,batch)
        if role=='L_s':
            if self._current_L_meta is not None:raise ValueError('Previous L metadata not consumed')
            self._current_L_meta=batch[3];self._current_L_labels=batch[1].tolist()
    def initialize(self,m,c,out):
        super().initialize(m,c,out)
        self.local_trace=gzip.open(self.out/'localatten_sequence.jsonl.gz','xt',encoding='utf-8',compresslevel=3)
    @contextmanager
    def native_scope(self,native):
        with BasicObserver.native_scope(self,native),F_optimizer_scope(self,'CENTER_UNITS'):
            factory=native.make_augmentor;native.make_augmentor=lambda cfg:NativeView(factory(cfg),self)
            try:yield
            finally:
                native.make_augmentor=factory
                if hasattr(self,'local_trace'):self.local_trace.close()
    def epoch(self,m,ctx,rows,out,device):
        if self._current_L_meta is not None or m._pair_primary is not None or m._pair_secondary is not None:
            raise ValueError('Local observation metadata/graphs escaped step')
        super().epoch(m,ctx,rows,out,device);epoch=len(rows);steps=epoch*222
        if m._supcon_calls!=steps or m._supcon_applied!=max(0,epoch-5)*222 or m._supcon_pending is not None or m._supcon_handles:
            raise ValueError('Original primary-pair SupCon schedule incomplete')
        if self.local.calls!=steps or self.local.packets!=steps*128:raise ValueError('One local observation per L slot required')
        self.local_trace.flush()
        self.epochs[-1]['SupCon']=dict(calls=m._supcon_calls,applied_calls=m._supcon_applied,last=m._supcon_records[-1],extra_model_forwards=0)
        self.epochs[-1]['local_attenuation']=self.local.summary();self.epochs[-1]['local_attenuation_seconds']=self.local_seconds
        d.write(self.out/'mechanism_progress.json',dict(records=self.epochs,head_fit_updates=0,U_truth_read=False,target_access=False))
    def finish(self):
        result=super().finish();m=self.m;stats=self.local.summary()
        if m._supcon_calls!=4440 or m._supcon_applied!=3330 or self.local.calls!=4440 or self.local.packets!=568320:
            raise ValueError('Full fixed E20 required')
        expected={(tx,rx,day,region) for tx in range(6) for rx in self.fit for day in (1,2,3) for region in range(16)}
        if len(self.local.ids)!=5040 or set(self.local_coverage)!=expected or set(self.local.starts)!=set(range(241)):
            raise ValueError('Complete physical/TX/RX/day/position support required')
        with gzip.open(self.out/'supcon_steps.jsonl.gz','xt',encoding='utf-8') as f:
            for r in m._supcon_records:f.write(json.dumps(r,allow_nan=False)+'\n')
        result['local_attenuation']=stats;result['local_attenuation_seconds']=self.local_seconds
        result['local_TX_RX_day_region_coverage']=[dict(TX=k[0],RX=k[1],day=k[2],region=k[3],count=v) for k,v in sorted(self.local_coverage.items())]
        result['supcon']=dict(calls=m._supcon_calls,applied_calls=m._supcon_applied,only_primary_pair=True,extra_model_forwards=0,new_parameters=0,U_truth_read=False)
        started=time.perf_counter();result['practical']=evaluate_practical(self)
        result['source_practical_seconds']=time.perf_counter()-started;result['source_practical_forward_packets']=32400
        d.write(self.out/'localatten_proof.json',dict(status='VERIFIED',local_attenuation=stats,pair_weights=[.5,.5],L_forwards_per_sample=4,
            all_L_views_supervised=True,same_generated_ordinary_IQ=True,SupCon_only_first_pair=True,U_recipe_unchanged=True,
            new_head=False,target_access=False,new_physical_channel=False,IQ_only_training_operator=True))
        d.write(self.out/'result.json',result);return result

def train(c):
    observer=Observer(c)
    view=SimpleNamespace(**{k:getattr(d,k) for k in ('validate','make_args','read','write','SOURCE','FULL_FP32_POLICY','ROOT','UPDATES')},
        source_validation=lambda model,loader,device:d.prior.old.old.source_validation(model,loader,device,fit=observer.fit))
    shared_train(c,design=view,model_api=api,observer=observer)
    return observer.finish()
