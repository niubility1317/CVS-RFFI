from contextlib import contextmanager
from pathlib import Path
import copy,torch
from . import design as d,model as api
from experiments.cvs_sigtime_locked20.source import Observer as OldObserver
from experiments.cvs_feature_sequence50.source import train as shared_train

class Observer:
    # The original RX1 exclusion, contract and call-local reader are unchanged.
    data=OldObserver.data
    source_contract=OldObserver.source_contract
    evaluate=OldObserver.evaluate
    def __init__(self,c):self.c=c
    def initialize(self,model,c,out):
        if c!=self.c:raise ValueError('One fixed source trajectory required')
        self.model=model;self.reader=copy.deepcopy(model).eval();self.out=Path(out)
        (self.out/'source_readouts').mkdir()
    @contextmanager
    def native_scope(self,native):yield
    def epoch(self,model,ctx,rows,out,device):
        epoch=len(rows)
        if model._feature_l_steps!=epoch*222:raise ValueError('Original222-step budget differs')
        d.write(self.out/'path_progress.json',dict(status='RUNNING',epoch=epoch,actual_steps=model._feature_l_steps,
            mode=self.c['arm'],target_access=False,U_truth_read=False,head_fit_updates=0))
    def finish(self):
        done=d.read(self.out/'completion.json')
        if done['optimizer_steps']!=4440 or done['initialization']!='scratch' or done['checkpoint_sources']:raise ValueError('Full scratch20 required')
        payload=torch.load(self.out/'final_ssdg.pth',map_location='cpu',weights_only=False)
        if any(not torch.equal(value,self.model.state_dict()[name].detach().cpu()) for name,value in payload['model'].items()):raise ValueError('Final live/checkpoint differ')
        value=self.evaluate(payload['model']);d.write(self.out/'source_readouts/FINAL.json',value)
        compact={key:v for key,v in value.items() if key not in ('physical_ids','labels','predictions')}
        result=dict(status='SOURCE_PATH_COMPLETE',mode=self.c['arm'],epoch=20,actual_steps=4440,
            source_health=done['final_training_L_CE']<=.5,final_training_L_CE=done['final_training_L_CE'],exports=dict(FINAL=compact),
            held_L_forward_packets=1260,smoke_forward_packets=2,additional_L_IQ_reads=1404,
            target_access=False,U_truth_read=False,head_fit_updates=0,default_head_joint_training=True,tail_average=False)
        d.write(self.out/'path_result.json',result);print('PATH_RESULT '+str(result),flush=True)
        return result
def train(c):
    observer=Observer(c);shared_train(c,design=d,model_api=api,observer=observer)
    return observer.finish()
