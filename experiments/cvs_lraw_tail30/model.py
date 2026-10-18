"""Exact originalP43 ORD128 hook;only native ordinary-L return IQ changes."""
from types import MethodType
from pathlib import Path
import torch
from . import design as d
from .protocol import check_checkpoint,check_source_metadata
from .schema import CONCAT_FIELDS,LRAW_FIELDS
from experiments.cvs_noise_shape_factorial43 import model as original
classifier,source_domain_count=original.classifier,original.source_domain_count

def hook(self,**kw):
 delta,previous_info=original.hook(self,**kw)
 record=self._full_records[-1];step=kw['optimizer_step'];ordinary=self._lraw_ordinary_records[-1]
 if ordinary['step']!=step or len(self._lraw_ordinary_records)!=step or record['crossview_mode']!='ORD128' or record['ordinary_raw_MSE']!=0. or record['L_forwards']!=6:raise ValueError('Exactly one L raw-return call and original six-L ORD128 hook required')
 state=dict(step=step,cumulative_step=4440+step,epoch=(step-1)//222+1,cumulative_epoch=20+(step-1)//222+1,ordinary_calls=1,ordinary_packets=128,discardedordinary_MSE=ordinary['discardedordinary_MSE'],discardedordinary_changed_packets=ordinary['discardedordinary_changed_packets'],returned_raw_MSE=ordinary['returned_raw_MSE'],full_native_draws=1,extra_NN=0,actual_ORD128_anchor_input='raw128 returned in ordinary slot',same_IQ_each_RDpair=True)
 self._lraw_records.append(state)
 info={'concat/'+k:record[k] for k in CONCAT_FIELDS}
 info.update({'lraw/'+k:state[k] for k in LRAW_FIELDS})
 return delta,info
hook._accepts_labeled_inputs=True

def original_config(c):
 d.validate(c)
 from experiments.cvs_noise_shape_factorial43 import design as b
 cfg=b.config(next(r for r in b.rows() if r['row_id']==c['row_id']),30);row=next(r for r in d.rows() if r['row_id']==c['row_id'])
 if cfg!=d.reference_config(row,30):raise ValueError('Actual pinned original recipe differs')
 return cfg

def attach(m,c):
 d.validate(c)
 if m._crossview_mode!='ORD128' or m._shape_As!=c['As'] or m._shape_mode!='NONE':raise ValueError('OriginalP43 ORD128/rawNONE/row As required')
 m._lraw_ordinary_records=[];m._lraw_records=[];m._source_labeled_feature_hook=MethodType(hook,m)
 m._feature_learning_report['LRaw']=dict(c['lraw'],parent=c['parent_source'],new_epochs=10,cumulative_epochs=30,pair_weights=[.5,.5,.25],SupCon='ORD128;actual ordinary anchors are raw128',per_step_L_NN=768,total_L_loss_scale=1.25,default_head_joint=True,extra_U_NN=0,no_new_parameters=True)
 return m

def install(m,c):return attach(original.install(m,original_config(c)),c)
def build(c,device='cpu'):return attach(original.build(original_config(c),device),c)
def own_model(c,device='cpu'):
 d.validate(c)
 from .source import parent_payload
 parent_payload(c);out=Path(c['output_root'])
 check_source_metadata(d.read(out/'completion.json'),d.read(out/'source_frozen.json'),d.read(out/'initialization.json'),d.read(out/'source_contract.json'),c,d.read(d.SOURCE)['role_ids'])
 cp=torch.load(out/'final_ssdg.pth',map_location='cpu',weights_only=False)
 check_checkpoint(cp,run=d.RUN,row=c['row_id'],epoch=10,updates=2220,baseline=c['parent_source']+'/final_ssdg.pth')
 m=build(c,device);m.load_state_dict(cp['model'],strict=True);return m.eval().requires_grad_(False)
