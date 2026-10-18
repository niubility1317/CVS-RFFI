"""Exact original P43 supervised L hook;only its scalar log uses the actual full6L record."""
from pathlib import Path
from types import MethodType
import torch
from . import design as d
from .protocol import check_checkpoint,check_source_metadata
from .schema import CONCAT_FIELDS
from experiments.cvs_noise_shape_factorial43 import model as original
from experiments.cvs_crossview_supcon41 import model as p41
classifier,source_domain_count=original.classifier,original.source_domain_count
capture_forward,terms,relations,supcon=p41.capture_forward,p41.terms,p41.relations,p41.supcon

def hook(self,**kw):
 delta,unused=original.hook(self,**kw)
 record=self._full_records[-1]
 return delta,{'concat/'+k:record[k] for k in CONCAT_FIELDS}
hook._accepts_labeled_inputs=True

def original_config(c):
 d.validate(c)
 from experiments.cvs_noise_shape_factorial43 import design as b
 cfg=b.config(next(r for r in b.rows() if r['row_id']==c['reference_row_id']),30)
 if cfg!=d.reference_config(next(r for r in d.rows() if r['row_id']==c['row_id']),30):raise ValueError('Actual pinned original B01 recipe differs')
 return cfg

def attach(m,c):
 d.validate(c)
 if m._crossview_mode!='ORD128' or m._shape_As!=c['As'] or m._shape_mode!='NONE':raise ValueError('OriginalP43 ORD128/As/rawNONE required')
 m._source_labeled_feature_hook=MethodType(hook,m)
 m._feature_learning_report['u_practical']=dict(c['u_practical'],parent=c['parent_source'],L_original_hook=True,no_new_parameters=True)
 return m

def install(m,c):return attach(original.install(m,original_config(c)),c)
def build(c,device='cpu'):return attach(original.build(original_config(c),device),c)
def own_model(c,device='cpu'):
 d.validate(c)
 from .source import parent_payload
 parent_payload(c);out=Path(c['output_root']);check_source_metadata(d.read(out/'completion.json'),d.read(out/'source_frozen.json'),d.read(out/'initialization.json'),d.read(out/'source_contract.json'),c,d.read(d.SOURCE)['role_ids'])
 cp=torch.load(out/'final_ssdg.pth',map_location='cpu',weights_only=False);check_checkpoint(cp,run=d.RUN,row=c['row_id'],epoch=10,updates=2220,baseline=c['parent_source']+'/final_ssdg.pth')
 if cp['args']['pseudo_strong_agreement']:raise ValueError('Both child rows require common teacher-only eligibility')
 m=build(c,device);m.load_state_dict(cp['model'],strict=True);return m.eval().requires_grad_(False)
