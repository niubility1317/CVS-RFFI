"""Unchanged original six-forward CE/RDrop/SupCon;only As IQ is erased."""
from pathlib import Path
import torch
from . import design as d
from .channel import MaskedPractical
from .protocol import check_checkpoint,check_source_metadata
from experiments.cvs_noise_shape_factorial43 import model as original
classifier,source_domain_count=original.classifier,original.source_domain_count
def attach(m,c):
 d.validate(c)
 if m._crossview_mode!='ORD128' or not m._shape_As or m._shape_mode!='NONE':raise ValueError('Original As/ORD128/raw control required')
 m._full_channel=MaskedPractical(m._full_channel,c['arm'])
 m._feature_learning_report['As_prefix81']=dict(c['masking'],parent=c['parent_source'],new_epochs=10,cumulative_epochs=30,default_head_joint=True)
 return m
def install(m,c):return attach(original.install(m,d.reference_config()),c)
def build(c,device='cpu'):return attach(original.build(d.reference_config(),device),c)
def own_model(c,device='cpu'):
 d.validate(c)
 from .source import parent_payload
 parent_payload(c);out=Path(c['output_root'])
 check_source_metadata(d.read(out/'completion.json'),d.read(out/'source_frozen.json'),d.read(out/'initialization.json'),d.read(out/'source_contract.json'),c,d.read(d.SOURCE)['role_ids'])
 cp=torch.load(out/'final_ssdg.pth',map_location='cpu',weights_only=False);check_checkpoint(cp,run=d.RUN,row=c['row_id'],epoch=10,updates=2220,baseline=c['parent_source']+'/final_ssdg.pth')
 m=build(c,device);m.load_state_dict(cp['model'],strict=True);return m.eval().requires_grad_(False)
