"""One native-shaped source step validates observer and own-EMA accounting."""
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import torch
from torch.nn import functional as F
from . import train_design as d,train_model as api
from .train_observer import Observer


def run():
    torch.set_num_threads(2);c=d.config(next(r for r in d.rows() if r['arm']=='B00'))
    m=api.build(c,'cpu');m.eval()
    with torch.no_grad():m(torch.zeros(2,2,256))
    native=SimpleNamespace(_detach_log_mapping=lambda values:values,deepcopy=copy.deepcopy)
    original=native._detach_log_mapping
    x=torch.randn(128,2,256);y=torch.arange(128)%6
    meta=(None,dict(rx_i=torch.ones(128,dtype=torch.long),day_i=torch.ones(128,dtype=torch.long),
                    eq_i=torch.ones(128,dtype=torch.long),base_index=torch.arange(128)))
    with tempfile.TemporaryDirectory() as tmp:
        out=Path(tmp);obs=Observer(c);obs.initialize(m,c,out)
        try:
            with obs.native_scope(native):
                teacher=native.deepcopy(m).eval().requires_grad_(False)
                if obs.teacher is not teacher or teacher.id_backbone.encoder._mixed_eval_packets:
                    raise ValueError('Native ownEMA creation not captured/reset')
                m.train()
                with m._phase1_labeled_feature_context():value=m(x,y_tx=y,return_aux=True)
                ce=F.cross_entropy(value['tx_logits'],y,label_smoothing=.01)
                correction,metrics=m._source_labeled_feature_hook(model=m,labeled_output=value,labels=y,extra=meta,
                    domains=None,optimizer_step=1,identity_loss=ce,labeled_count=128)
                native._detach_log_mapping(dict(metrics,**{'train/loss':float((ce+correction).detach())}))
                with torch.no_grad():
                    teacher(torch.zeros(2,2,256));m.eval();m(torch.zeros(2,2,256))
            if obs.steps!=1 or m.id_backbone.encoder._mixed_train_packets!=768 or m.id_backbone.encoder._mixed_eval_packets!=4 or teacher.id_backbone.encoder._mixed_eval_packets!=2:
                raise ValueError('Student L/eval and ownEMA forward accounting differs')
            if native._detach_log_mapping is not original or native.deepcopy is not copy.deepcopy:raise ValueError('Native observer scope was not restored')
        finally:obs.trace.close()
    result=dict(status='PASS',observer_step=1,synthetic_forward_packets=774,real_IQ_reads=0,checkpoint_loads=0,optimizer_steps=0)
    print(json.dumps(result));return result
