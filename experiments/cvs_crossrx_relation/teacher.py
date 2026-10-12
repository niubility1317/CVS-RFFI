"""Exact source-only new noCFO CE final20 teachers, no target score reads."""
from pathlib import Path
import torch
from . import design as d
from experiments.cvs_nocfo_foundation import design as f

def teacher_config(seed):
    if seed not in d.SEEDS:raise ValueError('Unregistered source teacher')
    return f.config(next(r for r in f.rows() if r['model_seed']==seed))

def expected_proof(seed):
    c=teacher_config(seed)
    return dict(status='VERIFIED',seed=seed,checkpoint=(Path(c['output_root'])/'final_ssdg.pth').as_posix(),epoch=20,optimizer_steps=4440,scratch_only=True,ancestors=[],source_roles='EXACT_MATCH',training_target_access=False,auxiliary_inheritance=False,execution_commit=d.TEACHER_EXECUTION_COMMIT,source_run=f.RUN,checkpoint_selection='final_only',explicit_CFO_identity_input=False,source_config=c,source_contract_ref=d.SOURCE,selection_context='all three preregistered final20 CE teachers fixed before scores; no target-driven choice')

def check_proof(value,seed):
    wanted=expected_proof(seed)
    if value!=wanted:raise ValueError('NoCFO CE source teacher provenance differs')
    return value

def verify(seed):
    from experiments.cvs_nocfo_foundation.evaluate import source_provenance,checkpoint_provenance
    c=teacher_config(seed);source=source_provenance(c);resolved=f.read(source/'resolved_config.json')
    if resolved['commit']!=d.TEACHER_EXECUTION_COMMIT or resolved['execution']['explicit_CFO_identity_input'] or resolved['execution']['U_iterator'] or resolved['execution']['teacher_source']!='none':raise ValueError('Teacher execution differs')
    payload=torch.load(source/'final_ssdg.pth',map_location='cpu',weights_only=False);checkpoint_provenance(c,payload)
    return check_proof(expected_proof(seed),seed)

def load(seed,device):
    proof=verify(seed)
    from experiments.cvs_nocfo_foundation.model import build
    model=build(teacher_config(seed),device,Path(proof['checkpoint'])).eval();model.requires_grad_(False)
    if model._identity_input_policy['explicit_CFO_identity_input']:raise ValueError('Explicit CFO teacher')
    return model,proof
