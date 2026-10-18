"""Delegate originalP43 six L forwards and ORD128;no new neural computation."""
from types import MethodType
from pathlib import Path
import torch
from . import design as d
from .protocol import check_checkpoint,check_source_metadata
from .schema import CONCAT_FIELDS,NO_U_FIELDS
from experiments.cvs_noise_shape_factorial43 import model as original
classifier,source_domain_count=original.classifier,original.source_domain_count

def hook(self,**kw):
 delta,previous_info=original.hook(self,**kw)
 record=self._full_records[-1];step=kw['optimizer_step']
 if record['crossview_mode']!='ORD128' or record['L_forwards']!=6:raise ValueError('Original six L forwards/ORD128 required')
 state=dict(step=step,cumulative_step=4440+step,epoch=(step-1)//222+1,cumulative_epoch=20+(step-1)//222+1,U_packets=0,U_student_NN=0,teacher_train_NN=0,teacher_eval_NN=0,EMA_updates=0,L_forwards=6,total_L_scale=1.25)
 self._no_u_records.append(state)
 info={'concat/'+k:record[k] for k in CONCAT_FIELDS}
 info.update({'no_u/'+k:state[k] for k in NO_U_FIELDS})
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
 m._no_u_records=[];m._source_labeled_feature_hook=MethodType(hook,m)
 m._feature_learning_report['tail_no_u']=dict(c['tail_no_u'],parent=c['parent_source'],new_epochs=10,cumulative_epochs=30,pair_weights=[.5,.5,.25],SupCon='unchanged ORD128',per_step_L_NN=768,total_L_loss_scale=1.25,default_head_joint=True,no_new_parameters=True)
 return m

def install(m,c):return attach(original.install(m,original_config(c)),c)
def build(c,device='cpu'):return attach(original.build(original_config(c),device),c)
def own_model(c,device='cpu'):
 d.validate(c)
 from .source import parent_payload
 parent_payload(c);out=Path(c['output_root'])
 check_source_metadata(d.read(out/'completion.json'),d.read(out/'source_frozen.json'),d.read(out/'initialization.json'),d.read(out/'source_contract.json'),c,d.read(d.SOURCE)['role_ids'])
 cp=torch.load(out/'final_ssdg.pth',map_location='cpu',weights_only=False)
 check_checkpoint(cp,run=d.RUN,row=c['row_id'],epoch=10,updates=2220,baseline=c['parent_source']+'/final_ssdg.pth',child=True)
 m=build(c,device);m.load_state_dict(cp['model'],strict=True);return m.eval().requires_grad_(False)
