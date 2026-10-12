from pathlib import Path
import torch
from . import design as d
from experiments.cvs_rx_heads import design as parent
from experiments.cvs_rx_heads.model import build,groups
from experiments.cvs_phase1_stack.recover import verify_contract
def own(c,L,device):
 parent.validate(c);p=Path(c['output']);v=d.read(p/'completion.json');resolved=d.read(p/'resolved_config.json');contract=d.read(p/'source_contract.json');verify_contract(contract,d.read(d.SOURCE));fit=[L['ids'][i] for i in (L['rx']!=c['held_rx']).nonzero().flatten().tolist()]
 if v['config']!=c or v['status']!='SOURCE_DIAGNOSTIC_COMPLETE' or v['identity_updates']!=6660 or len(v['epochs'])!=30 or v['fit_L_count']!=5040 or v['held_V_count']!=5400 or v['target_access'] or v['U_opened'] or not v['parameters_buffers_unchanged'] or v['parameters']!=180865 or v['head_partition_counts']!=[1260]*4 or resolved['checkpoint_loaded'] or resolved['ancestors'] or resolved['identity_input_policy']['explicit_CFO_identity_input']:raise ValueError('Sourceparent actualprovenance differs')
 ck=torch.load(p/'identity.pth',map_location='cpu',weights_only=False)
 if ck['config']!=c or ck['epoch']!=30 or ck['updates']!=6660 or ck['fit_ids']!=fit or ck['ancestors'] or ck['target_access'] or ck['held_rx']!=c['held_rx']:raise ValueError('Sourceparent checkpointfitroles differs')
 model=build(c,L['x'][L['rx']!=c['held_rx']],device);groups(model,c);model.load_state_dict(ck['model'],strict=True);return model.eval().requires_grad_(False)
