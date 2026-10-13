import time
from pathlib import Path
from . import design as d
COUNT=168000
def source_proof(c):
 d.validate_member(c);p=Path(c['output']);done=d.read(p/'completion.json');resolved=d.read(p/'resolved_config.json');frozen=d.read(p/'source_frozen.json');expect=710400 if c['source_U'] else 0
 if done['status']!='SOURCE_TRAINED' or done['config']!=c or done['epoch']!=30 or done['updates']!=6660 or done['logged_steps']!=6660 or done['target_access'] or done['U_truth_read'] or done['ancestors'] or done['V_forward'] or done['source_performance_gate'] or done['counts']['L_packets']!=852480 or done['counts']['U_packets']!=expect or done['counts']['identity_updates']!=6660 or done['distinct_U']!=(56700 if c['source_U'] else 0) or done['readouts']!=list(d.readouts(c)):raise ValueError('Reportmechanism actual training provenance differs')
 if frozen!=dict(status='SOURCE_FROZEN',config=c,readouts=list(d.readouts(c)),epoch=30,updates=6660,selection='fixed_final_epoch30',target_access=False) or resolved['config']!=c or resolved['checkpoint_loaded'] or resolved['ancestors'] or resolved['target_access'] or resolved['U_truth_read'] or resolved['V_forward'] or resolved['source_performance_gate'] or resolved['identity_input_policy']['explicit_CFO_identity_input'] or resolved['classifier']!=c['classifier'] or resolved['actual_initialization_seed']!=c['member_init_seed']:raise ValueError('Reportmechanism freeze differs')
 from experiments.cvs_phase1_stack.recover import verify_contract
 verify_contract(d.read(p/'source_contract.json'),d.read(d.SOURCE));return p
def own_model(c,name,device):
 import torch
 from .model import build
 p=source_proof(c)
 if name not in d.readouts(c):raise ValueError('Unregistered readout')
 ck=torch.load(p/(name+'.pth'),map_location='cpu',weights_only=False);contract=d.read(p/'source_contract.json')
 if ck['config']!=c or ck['readout']!=name or ck['epoch']!=30 or ck['updates']!=6660 or ck['ancestors'] or ck['initialization']!='scratch' or ck['actual_initialization_seed']!=c['member_init_seed'] or ck['target_access'] or ck['U_truth_read'] or len(ck['fit_ids'])!=6300 or sorted(ck['fit_ids'])!=sorted(contract['role_ids']['L_s']) or sorted(ck['fit_U_ids'])!=(sorted(contract['role_ids']['U_s']) if c['source_U'] else []):raise ValueError('Own final30 actualcheckpoint/data differs')
 model=build(c,device);model.load_state_dict(ck['model'],strict=True);return model.eval().requires_grad_(False)
