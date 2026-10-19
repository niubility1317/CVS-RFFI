from pathlib import Path
import torch
from . import design as d
from experiments.cvs_same_as_kd85 import model as old
classifier,source_domain_count,TEACHERS=old.classifier,old.source_domain_count,old.TEACHERS
def ref(c):return d.reference_config(next(r for r in d.rows() if r['row_id']==c['row_id']),c['segment'])
def install(m,c):d.validate(c);return old.install(m,ref(c))
def build(c,device='cpu'):
 d.validate(c);a=d.make_args(c,str(device));a.num_domains=source_domain_count(a);a.input_len=256
 torch.manual_seed(c['model_seed'])
 from experiments.cvs_phase1_stack.runtime import installed
 with installed(c) as native:m=native.build_baseline_model(a,torch.device(device))
 from experiments.cvs_source_identity_readout.input_policy import apply
 m._identity_input_policy=apply(m)
 return install(m,c).to(device)
def own_model(c,device='cpu'):
 d.validate(c)
 if c['segment']!=30:raise ValueError('Only own cumulative FINAL30')
 from .source import parent_payload
 parent_payload(c)
 p=Path(c['output_root']);done=d.read(p/'completion.json');frozen=d.read(p/'source_frozen.json');init=d.read(p/'initialization.json')
 if done['config']!=c or done['epoch']!=10 or done['optimizer_steps']!=2220 or done['target_access'] or frozen['config']!=c or frozen['selection']!='final_only' or frozen['target_access'] or init['target_contact'] or init['checkpoint_sources']!=c['checkpoint_sources'] or d.read(p/'source_contract.json')['role_ids']!=d.read(d.SOURCE)['role_ids']:raise ValueError('Own exactsource final provenance differs')
 cp=torch.load(p/'final_ssdg.pth',map_location='cpu',weights_only=False)
 if cp['epoch']!=10 or cp['a1_ema_successful_updates']!=2220 or cp['run_id']!=c['run_id'] or cp['candidate_id']!=c['row_id'] or cp['args']['baseline_ckpt']!=c['parent_source']+'/final_ssdg.pth' or cp['args']['teacher_ckpt']:raise ValueError('Own source final identity differs')
 m=build(c,device);m.load_state_dict(cp['model'],strict=True);return m.eval().requires_grad_(False)
