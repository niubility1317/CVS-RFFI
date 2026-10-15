"""Reuse the verified six-forward native hook, replacing only its IQ G callback."""
from functools import partial
from types import MethodType
import numpy as np
import torch
from . import response as r
from . import train_design as d
from experiments.cvs_mixed_iq20 import train_model as base
from experiments.cvs_mixed_iq20 import train_channel as channel

source_domain_count,classifier=base.source_domain_count,base.classifier
_bank=None
_pending=None
_original_proof=base.source_proof


def response_bank():
    global _bank
    if _bank is None:
        payload=d.read(d.ROOT/'source_bank_input.json')
        if (payload.get('source_run')!=d.config(d.rows()[0])['training_mechanism']['bank_source_run']
            or payload.get('primary')!='rectangular/DCT1to8' or payload.get('L_count')!=6300
            or any(payload.get(k) is not False for k in ('U_used','V_used','target_access'))):
            raise ValueError('Only fixed L-derived response bank permitted')
        _bank=r.bank(payload['predicted_contrasts'])
    return _bank


def generalization(x,meta,epoch,*,kind):
    global _pending
    keys=[(epoch,int(rx),int(day),int(eq),0) for rx,day,eq in zip(meta['rx_i'],meta['day_i'],meta['eq_i'])]
    h=r.kernels(response_bank(),kind,keys);out=r.apply_torch(x,h)
    power=x.square().sum(1).mean(1);error=float(((out.square().sum(1).mean(1)-power).abs()/power.clamp_min(1e-12)).max())
    if error>2e-6:raise ValueError('Response lost packet RMS')
    _pending=[dict(epoch=k[0],RX=k[1],day=k[2],eq=k[3],view=k[4],bank_index=r.session(*k)[0],strength=r.session(*k)[1]) for k in keys]
    return out,dict(max_relative_power_error=error,waveform_MSE=float((out-x).square().mean()))


def hook(self,**kwargs):
    global _pending
    result=self._response_base_hook(**kwargs)
    if _pending is None or len(_pending)!=128:raise ValueError('One generated G record per actual L sample required')
    labels=kwargs['labels'].detach().cpu().tolist()
    self._response_records=[dict(v,TX=int(tx)) for v,tx in zip(_pending,labels)];_pending=None
    return result


hook._accepts_labeled_inputs=False


def attach(m,c):
    kind='empirical' if c['arm']=='EMP' else 'rotated'
    channel.generalization=partial(generalization,kind=kind)
    m._response_base_hook=m._source_labeled_feature_hook
    m._source_labeled_feature_hook=MethodType(hook,m);m._response_records=[]
    m._feature_learning_report['source_response']=c['training_mechanism']['response_operator']
    return m


def install(m,c):return attach(base.install(m,c),c)
def build(c,device):return attach(base.build(c,device),c)


def source_proof(c):
    root=_original_proof(c);done=d.read(root/'response_coverage_complete.json')
    if done.get('status')!='VERIFIED' or done.get('packets')!=568320 or done.get('sessions')!=300 or not done.get('all_TX_each_session'):
        raise ValueError('Actual TX-independent G session coverage incomplete')
    return root


base.source_proof=source_proof
own_model=base.own_model
