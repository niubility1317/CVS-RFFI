"""Reuse LRaw30 equations;only cumulative clock/channel window extends to31..50."""
from pathlib import Path
from types import MethodType
import torch
from . import design as d
from .channel import Shifted,Practical
from .protocol import check_checkpoint,check_source_metadata
from experiments.cvs_lraw_tail30 import model as old
classifier,source_domain_count=old.classifier,old.source_domain_count

def hook(self,**kw):
 delta,info=old.hook(self,**kw)
 state=self._lraw_records[-1];step=kw['optimizer_step'];state.update(cumulative_step=6660+step,cumulative_epoch=30+(step-1)//222+1)
 return delta,info
hook._accepts_labeled_inputs=True

def attach(m,c):
 d.validate(c)
 m._full_channel=Shifted(Practical('train')) if c['As'] else old.original.NoAs();m._shape_offset=6660;m._source_labeled_feature_hook=MethodType(hook,m)
 m._feature_learning_report['LRaw']=dict(c['lraw'],parent=c['parent_source'],new_epochs=20,cumulative_epochs=50,epoch_offset=30,step_offset=6660,pair_weights=[.5,.5,.25],SupCon='ORD128;actual ordinary anchors are raw128',per_step_L_NN=768,total_L_loss_scale=1.25,default_head_joint=True,extra_U_NN=0,no_new_parameters=True)
 return m

def install(m,c):return attach(old.install(m,d.parent_config(c)),c)
def build(c,device='cpu'):return attach(old.build(d.parent_config(c),device),c)
def own_model(c,device='cpu'):
 d.validate(c)
 from .source import parent_payload
 parent_payload(c);out=Path(c['output_root'])
 done=d.read(out/'completion.json')
 check_source_metadata(done,d.read(out/'source_frozen.json'),d.read(out/'initialization.json'),d.read(out/'source_contract.json'),c,d.read(d.SOURCE)['role_ids'],20,4440)
 if done['cumulative_epochs']!=50 or done['cumulative_optimizer_steps']!=11100:raise ValueError('Only fixed cumulative FINAL50')
 cp=torch.load(out/'final_ssdg.pth',map_location='cpu',weights_only=False)
 check_checkpoint(cp,run=d.RUN,row=c['row_id'],epoch=20,updates=4440,baseline=c['parent_source']+'/final_ssdg.pth')
 m=build(c,device);m.load_state_dict(cp['model'],strict=True);return m.eval().requires_grad_(False)
