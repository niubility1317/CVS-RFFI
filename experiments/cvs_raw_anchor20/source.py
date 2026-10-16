from contextlib import contextmanager
import gzip,json
from types import SimpleNamespace
import torch
from . import design as d,model as api
from experiments.cvs_ordinary_fold20.source import Observer as OriginalObserver,shared_train
from .f_base.optimizer import native_scope as F_optimizer_scope

class NativeView:
    def __init__(self,inner,observer):object.__setattr__(self,'_inner',inner);object.__setattr__(self,'_observer',observer)
    def __getattr__(self,key):return getattr(self._inner,key)
    def __setattr__(self,key,value):setattr(self._inner,key,value)
    def __call__(self,x,*args,**kw):
        m=self._observer.m
        if m._anchor_G is not None or x.shape!=(128,2,256):raise ValueError('One fresh L view cache required')
        generated=self._inner(x,*args,**kw)
        m._anchor_G=generated.detach().clone();m._anchor_raw=x.detach().clone()
        return generated if self._observer.c['arm']=='A1' else x

class Observer(OriginalObserver):
    def __init__(self,c):
        super().__init__(c);self.optimizer=None
        prefix='id_backbone.encoder.'
        self.feature_parameter_names=[prefix+'core.id_backbone.cls_head.'+s for s in ('weight','gain')]
        self.feature_parameter_names += [prefix+f'sequence_pool.views.{i}.local_projection.weight' for i in range(3)]
        self.feature_parameter_names += [prefix+'core.id_backbone.sinc.'+s for s in ('low_coordinate','band_coordinate')]
    def initialize(self,m,c,out):
        super().initialize(m,c,out);self.model=m
    @contextmanager
    def native_scope(self,native):
        with super().native_scope(native),F_optimizer_scope(self,'CENTER_UNITS'):
            original=native.make_augmentor
            native.make_augmentor=lambda cfg:NativeView(original(cfg),self)
            try:yield
            finally:native.make_augmentor=original
    def epoch(self,m,ctx,rows,out,device):
        step=len(rows)*222
        if (self.optimizer is None or self.optimizer._sinc_unit_steps!=step or
            self.optimizer._sinc_physical_clip_calls!=step or len(self.optimizer._sinc_unit_lr_records)!=step):
            raise ValueError('Every F update requires its original physical clip and unit mapping')
        if (m._anchor_calls!=step or m._anchor_extra_packets!=step*256 or m._anchor_G is not None or m._anchor_raw is not None
            or m.id_backbone.encoder._ordinary_train_packets!=step*512+self.observed.get('U_s',0)):
            raise ValueError('Actual four-forward pair and cache consumption mismatch')
        super().epoch(m,ctx,rows,out,device)
        self.epochs[-1]['raw_anchor']=dict(calls=m._anchor_calls,L_forwards=step*512,extra_second_pair_packets=m._anchor_extra_packets,
            first_view=self.c['training_mechanism']['first_pair'],second_view=self.c['training_mechanism']['second_pair'],pair_weights=[.5,.5])
        self.epochs[-1]['F_optimizer']=dict(mode='CENTER_UNITS',steps=step,physical_clip_calls=step,
            last_actual_mapping=self.optimizer._sinc_unit_lr_records[-1])
        d.write(self.out/'mechanism_progress.json',dict(records=self.epochs,head_fit_updates=0,U_truth_read=False,target_access=False))
    def finish(self):
        result=super().finish()
        if self.optimizer._sinc_unit_steps!=4440 or self.optimizer._sinc_physical_clip_calls!=4440:
            raise ValueError('Incomplete original F optimizer trajectory')
        with gzip.open(self.out/'F_optimizer_steps.jsonl.gz','xt',encoding='utf-8',compresslevel=3) as f:
            for row in self.optimizer._sinc_unit_lr_records:f.write(json.dumps(row,allow_nan=False)+'\n')
        d.write(self.out/'F_implementation_proof.json',dict(status='VERIFIED',feature='SIG_TIME_X3',
            coordinates='CENTER_25K',optimizer='CENTER_UNITS',steps=4440,physical_clip_calls=4440,
            q_scale_Hz=25000.,sample_rate_Hz=25000000.,new_models_inherited=0,
            default_head_joint=True,head_fit_updates=0,source_only=True,target_access=False))
        return result

def train(c):
    observer=Observer(c)
    view=SimpleNamespace(**{k:getattr(d,k) for k in ('validate','make_args','read','write','SOURCE','FULL_FP32_POLICY','ROOT','UPDATES')},
        source_validation=lambda model,loader,device:d.old.source_validation(model,loader,device,fit=observer.fit))
    shared_train(c,design=view,model_api=api,observer=observer)
    result=observer.finish()
    d.write(c['output_root']+'/raw_anchor_proof.json',dict(actual_L_forwards_per_sample=4,actual_extra_second_pair_packets=1136640,
        ordinary_draws_per_step=int(c['training_mechanism']['ordinary_L']),native_first_view=c['training_mechanism']['first_pair'],
        second_pair_view=c['training_mechanism']['second_pair'],pair_weights=[.5,.5],cross_view_consistency=False,target_access=False))
    return result
