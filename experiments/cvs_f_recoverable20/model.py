from types import MethodType
import torch
from torch.nn import functional as F
from . import design as d
from experiments.cvs_f_crossrx_folds20.model import hook as paired_hook
from experiments.cvs_f_crossrx_practical20.supcon import attach as attach_supcon
from experiments.cvs_raw_anchor20 import model as raw
source_domain_count,classifier=raw.source_domain_count,raw.classifier
def hook(self,**kw):
    step=kw['optimizer_step'];y=kw['labels'];logits=kw['labeled_output']['tx_logits'];meta=kw['extra'][1]
    if step!=self._course_calls+1 or y.shape!=(128,) or logits.shape!=(128,6) or not bool(((y>=0)&(y<6)).all()):raise ValueError('Only ordered realL128 default-head progress')
    with torch.no_grad():
        pred=logits.argmax(1);margin=(logits.gather(1,y[:,None]).flatten()-logits.masked_fill(F.one_hot(y,6).bool(),-torch.inf).max(1).values)/30
        ce=F.cross_entropy(logits,y,label_smoothing=.01,reduction='none');tx=y.tolist();rx=meta['rx_i'].tolist();day=meta['day_i'].tolist();correct=(pred==y).tolist();mm=margin.tolist();cc=ce.tolist()
    for t,r,dd,ok,ma,loss in zip(tx,rx,day,correct,mm,cc):
        if r not in self._course_fit or dd not in (1,2,3):raise ValueError('Only fitL progress cells')
        a=self._course_cells.setdefault((t,r,dd),[0,0,0.,0.]);a[0]+=1;a[1]+=int(ok);a[2]+=ma;a[3]+=loss
    self._course_calls=step
    return paired_hook(self,**kw)
hook._accepts_labeled_inputs=False
def attach(m,c):
    m=raw.attach(m,c,validator=d.validate);m._source_labeled_feature_hook=MethodType(hook,m);m._pair_primary=None;m._pair_secondary=None
    m._course_calls=0;m._course_cells={};m._course_fit=tuple(c['fold']['fit_RX']);m._feature_learning_report['baseline_CROSS_RX']=c['supcon']
    return attach_supcon(m,c)
def install(m,c):return attach(raw.F_configure(raw.base.legacy.install(m,d.parent_config()),c),c)
def build(c,device='cpu'):return attach(raw.F_configure(raw.base.legacy.build(d.parent_config(),'cpu'),c).to(device),c)
