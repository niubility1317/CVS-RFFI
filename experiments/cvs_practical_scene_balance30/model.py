"""Exact originalP43 ORD128 hook;only its existingAs64 channel assignment changes."""
from types import MethodType
from pathlib import Path
import torch
from . import design as d
from .protocol import check_checkpoint,check_source_metadata
from .channel import Shifted,Practical
from .schema import CONCAT_FIELDS,BALANCE_FIELDS
from experiments.cvs_noise_shape_factorial43 import model as original
classifier,source_domain_count=original.classifier,original.source_domain_count

def hook(self,**kw):
 delta,previous_info=original.hook(self,**kw)
 record=self._full_records[-1];balance=self._scene_balance_records[-1];step=kw['optimizer_step'];mask=(torch.arange(128,device=kw['labels'].device)+step).remainder(2)!=0
 balance['selected_L_positions']=mask.nonzero(as_tuple=False).flatten().tolist();balance['TX']=kw['labels'][mask].tolist()
 if balance['step']!=step or record['crossview_mode']!='ORD128' or balance['scene_counts']!=record['scene_counts']:raise ValueError('Same originalAs64 call and ORD128 hook required')
 # OriginalP43 already adds the raw pair. Native total_L is normalized to full6L here.
 info={'concat/'+k:record[k] for k in CONCAT_FIELDS}
 info.update({'balance/'+k:balance[k] for k in BALANCE_FIELDS})
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
 if m._crossview_mode!='ORD128' or not m._shape_As or m._shape_mode!='NONE':raise ValueError('OriginalP43 ORD128/As/rawNONE required')
 m._full_channel=Shifted(Practical('train'));m._scene_balance_records=m._full_channel.scene_balance_records;m._source_labeled_feature_hook=MethodType(hook,m)
 m._feature_learning_report['sceneBalance']=dict(c['scene_balance'],parent=c['parent_source'],new_epochs=10,cumulative_epochs=30,pair_weights=[.5,.5,.25],SupCon='unchanged ORD128',per_step_L_NN=768,total_L_loss_scale=1.25,default_head_joint=True,extra_U_NN=0,no_new_parameters=True)
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
