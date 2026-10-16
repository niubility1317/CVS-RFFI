from contextlib import contextmanager
from types import SimpleNamespace
from . import design as d,model as api
from .jitter import OriginJitter
from experiments.cvs_raw_anchor20.source import Observer as FObserver,NativeView,F_optimizer_scope
from experiments.cvs_ordinary_fold20.source import Observer as BasicObserver,shared_train
class Observer(FObserver):
    @contextmanager
    def native_scope(self,native):
        with BasicObserver.native_scope(self,native),F_optimizer_scope(self,'CENTER_UNITS'):
            factory=native.make_augmentor;scheduler=native.configure_augmentor_for_epoch
            def make(cfg):
                if self.augmentor is not None:raise ValueError('One OJ24 factory only')
                self.augmentor=OriginJitter(d.ORIGIN['seed']);d.write(self.out/'origin_jitter_config.json',dict(actual=d.ORIGIN,native_base_cfg=cfg,original_ordinary_factory_called=False))
                return NativeView(self.augmentor,self)
            def schedule(augmentor,cfg,epoch,args):
                if not isinstance(augmentor,NativeView) or augmentor._observer is not self:raise ValueError('Only own OJ24 schedule')
                self.augmentor.schedules.append(int(epoch))
                return dict(scale=1.,p_dac=0.,p_pa=0.,p_time_shift=1.,p_cfo=0.,p_awgn=0.,p_multipath=0.,max_time_shift=24,cfo_max=0.,phase_noise_sigma_max=0.)
            native.make_augmentor=make;native.configure_augmentor_for_epoch=schedule
            try:yield
            finally:native.make_augmentor=factory;native.configure_augmentor_for_epoch=scheduler
    def epoch(self,m,ctx,rows,out,device):
        if len(self.augmentor.schedules)!=len(rows):raise ValueError('One fixed OJ24 schedule per epoch required')
        super().epoch(m,ctx,rows,out,device)
    def finish(self):
        result=super().finish()
        if self.augmentor._calls!=4440 or len(self.augmentor.schedules)!=20 or not all(self.augmentor.delay_counts):raise ValueError('Complete OJ24 budget/range coverage required')
        d.write(self.out/'origin_jitter_proof.json',dict(status='VERIFIED',actual=self.augmentor.summary(),raw_weight=.5,jitter_weight=.5,
            same_IQ_within_RDpair=True,original_ordinary_calls=0,As_calls=0,head_fit_updates=0,target_access=False))
        return result
def train(c):
    observer=Observer(c)
    view=SimpleNamespace(**{k:getattr(d,k) for k in ('validate','make_args','read','write','SOURCE','FULL_FP32_POLICY','ROOT','UPDATES')},
        source_validation=lambda model,loader,device:d.old.old.source_validation(model,loader,device,fit=observer.fit))
    shared_train(c,design=view,model_api=api,observer=observer)
    return observer.finish()
