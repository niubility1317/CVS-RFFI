"""Strict own final source state and original jointly trained default classifier."""
from pathlib import Path
from . import design as d
def own_model(c,device='cpu'):
 import torch
 from .binding import modules
 r=d.row(c['row_id']);b,api=modules(r)
 if c!=b['config'](30):raise ValueError('Only registered final30 configuration')
 p=Path(c['output_root']);done=d.read(p/'completion.json');init=d.read(p/'initialization.json');frozen=d.read(p/'source_frozen.json')
 if done['status']!='SOURCE_TRAINED' or done['config']!=c or done['epoch']!=10 or done['optimizer_steps']!=2220 or done['target_access'] or done['head_fit_updates']:raise ValueError('Incomplete or illegal source state')
 if frozen['config']!=c or frozen['target_access'] or init['target_contact'] or init['seed']!=r['model_seed']:raise ValueError('Final freeze or actual seed differs')
 if d.read(p/'source_contract.json')['role_ids']!=d.read(d.SOURCE)['role_ids']:raise ValueError('CHECKPOINT_DATA_CONTRACT_MISMATCH')
 parent=Path(c['parent_source']);v=d.read(parent/'completion.json');initial=d.read(parent/'initialization.json')
 if v['epoch']!=20 or v['optimizer_steps']!=4440 or v['target_access'] or not initial['scratch_only'] or initial['ancestors'] or initial['checkpoint_sources'] or initial['seed']!=r['model_seed'] or initial['target_contact']:raise ValueError('Own parent lineage differs')
 if d.read(parent/'source_contract.json')['role_ids']!=d.read(d.SOURCE)['role_ids']:raise ValueError('Parent physical roles differ')
 cp=torch.load(p/'final_ssdg.pth',map_location='cpu',weights_only=False)
 if cp['epoch']!=10 or cp['a1_ema_successful_updates']!=2220 or cp['args']['seed']!=r['model_seed'] or cp['run_id']!=d.RUN or cp['candidate_id']!=r['row_id'] or cp['args']['baseline_ckpt']!=str(parent/'final_ssdg.pth') or cp['args']['teacher_ckpt']:raise ValueError('Final checkpoint provenance differs')
 model=api.build(c,device);model.load_state_dict(cp['model'],strict=True)
 if model._selected_factory_seed!=r['model_seed']:raise ValueError('Actual factory seed differs')
 return model.eval().requires_grad_(False)
