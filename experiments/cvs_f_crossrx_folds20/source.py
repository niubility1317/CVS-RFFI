from collections import Counter
from contextlib import contextmanager
import copy,gzip,json,time
from pathlib import Path
from types import SimpleNamespace
import torch
from torch.nn import functional as F
from . import design as d,model as api
from .channel import Practical
from experiments.cvs_ordinary_fold20.source import Observer as BasicObserver,shared_train
from experiments.cvs_ordinary_fold20.design import source_validation
from experiments.cvs_raw_anchor20.source import Observer as FObserver,F_optimizer_scope

class NativeView:
    def __init__(self,inner,observer):object.__setattr__(self,'_inner',inner);object.__setattr__(self,'_observer',observer)
    def __getattr__(self,k):return getattr(self._inner,k)
    def __setattr__(self,k,v):setattr(self._inner,k,v)
    def __call__(self,x,*args,**kw):
        o=self._observer;m=o.m
        if m._pair_primary is not None or m._pair_secondary is not None or o._current_L_meta is None:raise ValueError('One physical L cache and paired view required')
        meta,labels=o._current_L_meta,o._current_L_labels;o._current_L_meta=o._current_L_labels=None
        primary=self._inner(x,*args,**kw);step=m._anchor_calls+1
        if o.c['As']:
            secondary,records=o.channel.apply(x,meta,(step-1)//222+1,step)
            for i,r in enumerate(records):o.channel_TX[(labels[i],int(meta['rx_i'][i]),r['scene'],r['state_end'])]+=1
        else:secondary=primary
        m._pair_primary=primary.detach().clone();m._pair_secondary=secondary.detach().clone()
        return primary

class Observer(FObserver):
    def __init__(self,c):
        super().__init__(c);self._current_L_meta=None;self._current_L_labels=None;self.channel_TX=Counter();self.channel=None
    def record(self,role,batch):
        super().record(role,batch)
        if role=='L_s':
            if self._current_L_meta is not None:raise ValueError('Previous L physical metadata not consumed')
            self._current_L_meta=batch[3];self._current_L_labels=batch[1].tolist()
    @contextmanager
    def native_scope(self,native):
        with BasicObserver.native_scope(self,native),F_optimizer_scope(self,'CENTER_UNITS'):
            factory=native.make_augmentor
            native.make_augmentor=lambda cfg:NativeView(factory(cfg),self)
            try:yield
            finally:native.make_augmentor=factory
    def initialize(self,m,c,out):
        super().initialize(m,c,out);self.channel=Practical('train',self.held);d.write(self.out/'practical_channel_audit.json',self.channel.audit())
    def epoch(self,m,ctx,rows,out,device):
        if self._current_L_meta is not None or m._pair_primary is not None or m._pair_secondary is not None:raise ValueError('Paired metadata/graphs escaped a step')
        super().epoch(m,ctx,rows,out,device);epoch=len(rows);steps=epoch*222
        if m._supcon_calls!=steps or m._supcon_applied!=max(0,epoch-5)*222 or m._supcon_pending is not None or m._supcon_handles:raise ValueError('Original clean-pair SupCon E20 schedule incomplete')
        if self.channel.calls!=(steps if self.c['As'] else 0):raise ValueError('One actual practical generation per C1 update and zero C0 generations required')
        self.epochs[-1]['SupCon']=dict(calls=m._supcon_calls,applied_calls=m._supcon_applied,last=m._supcon_records[-1],extra_model_forwards=0)
        self.epochs[-1]['training_practical']=self.channel.summary()
        d.write(self.out/'mechanism_progress.json',dict(records=self.epochs,head_fit_updates=0,U_truth_read=False,target_access=False))
    def finish(self):
        result=super().finish();m=self.m
        if m._supcon_calls!=4440 or m._supcon_applied!=3330:raise ValueError('Full E20 contrastive schedule required')
        with gzip.open(self.out/'supcon_steps.jsonl.gz','xt',encoding='utf-8') as f:
            for r in m._supcon_records:f.write(json.dumps(r,allow_nan=False)+'\n')
        if self.c['As'] and {(t,r,s) for t,r,s,z in self.channel_TX}!={(t,r,s) for t in range(6) for r in self.fit for s in d.TRAIN_SCENES}:
            raise ValueError('Full TX x fit RX x training scene exposure required')
        if self.c['As'] and {(t,r,z) for t,r,s,z in self.channel_TX}!={(t,r,z) for t in range(6) for r in self.fit for z in ('LOS','PARTIAL','BLOCKED')}:
            raise ValueError('Full TX x fit RX x nominal state coverage required')
        result['training_practical']=self.channel.summary()
        result['TX_RX_scene_state_coverage']=[dict(TX=k[0],RX=k[1],scene=k[2],state=k[3],count=v) for k,v in sorted(self.channel_TX.items())]
        result['supcon']=dict(calls=m._supcon_calls,applied_calls=m._supcon_applied,only_primary_pair=True,extra_model_forwards=0,new_parameters=0,U_truth_read=False)
        started=time.perf_counter();result['practical']=evaluate_practical(self)
        result['source_practical_seconds']=time.perf_counter()-started
        result['source_practical_forward_packets']=6*5400
        d.write(self.out/'FcrossRX_pair_proof.json',dict(status='VERIFIED',SupCon=result['supcon'],training_practical=result['training_practical'],
            pair_weights=[.5,.5],primary_view='old ordinary clean',secondary_view=self.c['training_mechanism']['second_pair'],
            L_forwards_per_sample=4,all_L_views_supervised=True,student_confidence_gate=False,ordinary_before_satellite=False,
            U_recipe_unchanged=True,head_fit_updates=0,target_access=False))
        d.write(self.out/'result.json',result);return result

def evaluate_practical(o):
    generator=Practical('validation',o.held);reader=copy.deepcopy(o.m).eval().requires_grad_(False)
    before={k:v.detach().clone() for k,v in reader.state_dict().items()};saved={}
    for scene in d.SCENES:
        cm=torch.zeros(6,6,dtype=torch.long);ce=0.;count=0
        with gzip.open(o.out/(scene+'_source_predictions.jsonl.gz'),'xt',encoding='utf-8') as output,torch.no_grad():
            for x,y,domain,meta in o.held_loader:
                values,physics=generator.apply(x.to(next(reader.parameters()).device),meta,scene=scene)
                logits=reader.forward_identity_only(values)['tx_logits'];truth=y.to(logits.device);pred=logits.argmax(1)
                if logits.shape!=(len(y),6) or not bool(torch.isfinite(logits).all()):raise ValueError('Finite default-head source predictions required')
                ce+=float(F.cross_entropy(logits,truth,reduction='sum'))
                margins=(logits.gather(1,truth[:,None]).flatten()-logits.masked_fill(F.one_hot(truth,6).bool(),-torch.inf).max(1).values)/30
                for i in range(len(y)):
                    t,p=int(y[i]),int(pred[i]);cm[t,p]+=1;count+=1
                    output.write(json.dumps(dict(label=t,prediction=p,margin=float(margins[i]),
                        eq=int(meta['eq_i'][i]),RX=int(meta['rx_i'][i]),day=int(meta['day_i'][i]),sig_i=int(meta['sig_i'][i]),base_index=int(meta['base_index'][i]),physics=physics[i]),allow_nan=False)+'\n')
        if count!=5400:raise ValueError('Whole held source V required in every scene')
        saved[scene]=dict(count=count,correct=int(cm.diagonal().sum()),accuracy=int(cm.diagonal().sum())/count,
            macro_f1=float((2*cm.diagonal().double()/(cm.sum(0)+cm.sum(1)).clamp_min(1)).mean()),mean_CE=ce/count,confusion=cm.tolist())
    if any(not torch.equal(v,reader.state_dict()[k]) for k,v in before.items()):raise ValueError('Source V changed frozen model state')
    d.write(o.out/'source_practical_evaluation.json',dict(status='COMPLETE',scenes=saved,channel=generator.summary(),audit=generator.audit(),
        packet_forwards=generator.packets,optimizer_updates=0,EMA_updates=0,target_access=False))
    return saved

def train(c):
    observer=Observer(c)
    view=SimpleNamespace(**{k:getattr(d,k) for k in ('validate','make_args','read','write','SOURCE','FULL_FP32_POLICY','ROOT','UPDATES')},
        source_validation=lambda model,loader,device:source_validation(model,loader,device,fit=observer.fit))
    shared_train(c,design=view,model_api=api,observer=observer)
    return observer.finish()
