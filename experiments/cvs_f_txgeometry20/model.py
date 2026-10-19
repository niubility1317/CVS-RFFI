"""No new layers: exact F extractor and unchanged existing default head."""
from experiments.cvs_f_supcon50 import design as original,model as api
from . import design as d

def build(c,device):
 d.validate(c)
 m=api.build(original.config(next(r for r in original.training_rows() if r['arm']=='CROSS_RX' and r['model_seed']==c['model_seed'])),device)
 m._feature_learning_report['local_TXgeometry20']=c
 return m

def args(c):
 a=original.make_args(original.config(next(r for r in original.training_rows() if r['arm']=='CROSS_RX' and r['model_seed']==c['model_seed'])),'cuda:0')
 a.epochs=20;a.label_epochs=20;a.pseudo_epochs=0;a.use_ema_teacher=False;a.num_workers=0
 return a
