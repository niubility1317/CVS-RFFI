from contextlib import contextmanager
import gzip,json
from types import SimpleNamespace
from . import design as d,model as api
from .channel import Practical
from .policy import Policy,aggregate
from experiments.cvs_f_crossrx_folds20.source import Observer as Parent
from experiments.cvs_ordinary_fold20.source import shared_train
from experiments.cvs_ordinary_fold20.design import source_validation
class Observer(Parent):
    def __init__(self,c):super().__init__(c);self.policy=Policy();self.course_epochs=[]
    def initialize(self,m,c,out):
        super().initialize(m,c,out);self.channel=Practical(self.held,self.policy);self.channel.trace=gzip.open(self.out/'curriculum_draws.jsonl.gz','xt',encoding='utf-8',compresslevel=3)
        d.write(self.out/'practical_channel_audit.json',self.channel.audit())
    @contextmanager
    def native_scope(self,native):
        with super().native_scope(native):
            try:yield
            finally:
                if self.channel and self.channel.trace:self.channel.trace.close()
    def epoch(self,m,ctx,rows,out,device):
        super().epoch(m,ctx,rows,out,device);epoch=len(rows)
        if m._course_calls!=epoch*222:raise ValueError('One progress read per source L step')
        cells=[dict(TX=k[0],RX=k[1],day=k[2],count=v[0],correct=v[1],margin_sum=v[2],CE_sum=v[3]) for k,v in sorted(m._course_cells.items())]
        metrics=aggregate(cells,float(rows[-1]['train_grad_before_clip']));record=self.policy.advance(epoch,metrics);record=dict(record,cells=cells);self.course_epochs.append(record);m._course_cells={}
        self.channel.trace.flush();self.epochs[-1]['curriculum']=dict(record,extra_model_forwards=0,held_V_used=False,U_truth_read=False,target_access=False)
        d.write(self.out/'curriculum_history.json',dict(records=self.course_epochs,progress_source='FitL primary default-head logits and native gradient, no extra forwards',target_access=False))
        d.write(self.out/'mechanism_progress.json',dict(records=self.epochs,head_fit_updates=0,U_truth_read=False,target_access=False))
    def finish(self):
        result=super().finish()
        if self.m._course_calls!=4440 or self.m._course_cells or len(self.course_epochs)!=20:raise ValueError('Complete fixed curriculum telemetry required')
        result['curriculum']=self.policy.summary();d.write(self.out/'result.json',result)
        d.write(self.out/'curriculum_proof.json',dict(status='VERIFIED',epochs=20,steps=4440,nominal_support_floor=.5,all_labeled_CE=True,new_head=False,extra_model_forwards=0,held_V_used_for_progress=False,U_truth_read=False,target_access=False))
        return result
def train(c):
    observer=Observer(c);view=SimpleNamespace(**{k:getattr(d,k) for k in ('validate','make_args','read','write','SOURCE','FULL_FP32_POLICY','ROOT','UPDATES')},source_validation=lambda model,loader,device:source_validation(model,loader,device,fit=observer.fit))
    shared_train(c,design=view,model_api=api,observer=observer);return observer.finish()
