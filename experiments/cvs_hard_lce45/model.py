"""Fixed F/head/QoS and IQ path; only supervised-L CE smoothing changes."""
from types import MethodType
from pathlib import Path
import torch
from . import design as d
from .losses import hard_L_terms
from experiments.cvs_noise_shape_factorial43 import model as old
from experiments.cvs_crossview_supcon41 import model as shared
classifier,source_domain_count=old.classifier,old.source_domain_count

def hook(self,**kw):
 records=[];original=shared.terms
 with hard_L_terms(shared,records):delta,info=old.hook(self,**kw)
 if shared.terms is not original or len(records)!=3:raise ValueError('Only three existing L RD pairs;U function must be restored')
 weights=(.5,.5,.25);diff=sum(w*(r['CE0']-r['CE01']) for w,r in zip(weights,records))
 record=self._full_records[-1];record['L_label_smoothing']=0.;record['native_U_label_smoothing']=.01;record['loss_only_ablation']=True
 self._hard_records.append(dict(step=kw['optimizer_step'],pairs=records,weighted_CE0_minus_CE01=diff,L_total=record['total_L'],original_L01_counterfactual_value=record['total_L']-diff,U_function_restored=True,extra_NN=0))
 info['hardLCE/weighted_CE_difference']=diff
 return delta,info
hook._accepts_labeled_inputs=True

def attach(m,c):
 d.validate(c);m._hard_records=[];m._source_labeled_feature_hook=MethodType(hook,m)
 m._feature_learning_report['hardLCE45']=dict(L_smoothing=0.,native_U_smoothing=.01,new_Ag=False,new_losses=0,new_NN=0,original_U_unchanged=True)
 return m

def install(m,c):return attach(old.install(m,d.reference_config(c)),c)
def build(c,device='cpu'):return attach(old.build(d.reference_config(c),device),c)
def own_model(c,device='cpu'):
 d.validate(c)
 if c['segment']!=30:raise ValueError('Only own fixed FINAL30')
 from .source import parent_payload
 parent_payload(c)
 p=Path(c['output_root']);done=d.read(p/'completion.json');frozen=d.read(p/'source_frozen.json');init=d.read(p/'initialization.json')
 if done['config']!=c or done['epoch']!=10 or done['optimizer_steps']!=2220 or done['target_access'] or frozen['config']!=c or frozen['selection']!='final_only' or frozen['target_access'] or init['target_contact'] or init['checkpoint_sources']!=c['checkpoint_sources'] or d.read(p/'source_contract.json')['role_ids']!=d.read(d.SOURCE)['role_ids']:raise ValueError('Own source final provenance differs')
 cp=torch.load(p/'final_ssdg.pth',map_location='cpu',weights_only=False)
 if cp['epoch']!=10 or cp['a1_ema_successful_updates']!=2220 or cp['run_id']!=d.RUN or cp['candidate_id']!=c['row_id'] or cp['args']['baseline_ckpt']!=c['parent_source']+'/final_ssdg.pth' or cp['args']['teacher_ckpt'] or float(cp['args']['label_smoothing'])!=.01:raise ValueError('Own final/U source identity differs')
 m=build(c,device);m.load_state_dict(cp['model'],strict=True);return m.eval().requires_grad_(False)
