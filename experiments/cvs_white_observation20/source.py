from contextlib import contextmanager
from types import SimpleNamespace
from . import design as d,model as api
from .noise import WhiteObservation
from experiments.cvs_raw_anchor20.source import Observer as FObserver,NativeView,F_optimizer_scope
from experiments.cvs_ordinary_fold20.source import Observer as BasicObserver,shared_train

class Observer(FObserver):
    @contextmanager
    def native_scope(self,native):
        with BasicObserver.native_scope(self,native),F_optimizer_scope(self,'CENTER_UNITS'):
            factory=native.make_augmentor;scheduler=native.configure_augmentor_for_epoch
            def make(cfg):
                if self.augmentor is not None:raise ValueError('Only one white observation factory')
                self.augmentor=WhiteObservation(d.NOISE['seed'])
                d.write(self.out/'observation_noise_config.json',dict(actual=d.NOISE,native_base_cfg=cfg,whole_ordinary_factory_called=False))
                return NativeView(self.augmentor,self)
            def schedule(augmentor,cfg,epoch,args):
                if not isinstance(augmentor,NativeView) or augmentor._observer is not self:raise ValueError('Only own L observation schedule')
                self.augmentor.schedules.append(int(epoch))
                return dict(scale=1.,p_dac=0.,p_pa=0.,p_time_shift=0.,p_cfo=0.,p_awgn=1.,p_multipath=0.,max_time_shift=0,cfo_max=0.,phase_noise_sigma_max=0.)
            native.make_augmentor=make;native.configure_augmentor_for_epoch=schedule
            try:yield
            finally:native.make_augmentor=factory;native.configure_augmentor_for_epoch=scheduler
    def epoch(self,m,ctx,rows,out,device):
        if len(self.augmentor.schedules)!=len(rows):raise ValueError('One fixed white schedule per epoch required')
        super().epoch(m,ctx,rows,out,device)
    def finish(self):
        result=super().finish()
        if self.augmentor._calls!=4440 or len(self.augmentor.schedules)!=20:raise ValueError('Complete white observation budget required')
        d.write(self.out/'white_observation_proof.json',dict(status='VERIFIED',actual=self.augmentor.summary(),
            fixed_raw_weight=.5,white_weight=.5,same_IQ_within_each_RD_pair=True,
            whole_ordinary_calls=0,As_calls=0,head_fit_updates=0,target_access=False))
        return result
def train(c):
    observer=Observer(c)
    view=SimpleNamespace(**{k:getattr(d,k) for k in ('validate','make_args','read','write','SOURCE','FULL_FP32_POLICY','ROOT','UPDATES')},
        source_validation=lambda model,loader,device:d.old.old.source_validation(model,loader,device,fit=observer.fit))
    shared_train(c,design=view,model_api=api,observer=observer)
    return observer.finish()
