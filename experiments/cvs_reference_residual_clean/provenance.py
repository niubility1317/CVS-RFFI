"""Read-only source freeze and checkpoint lineage; no query access."""
from pathlib import Path
import torch
from experiments.cvs_reference_residual_clean.contracts import SOURCE_ROOT,SOURCE_COMMIT,SOURCE_RUN,SOURCE_CONTRACT,CLASSES,expected_rows,read,validate_config
from experiments.cvs_reference_residual_identity.dispatch import read_source_record
from experiments.cvs_reference_residual_identity.source import validate_config as validate_source
from experiments.cvs_reference_residual_identity.model import build
from experiments.cvs_reference_residual_identity.contracts import architecture_contract as dual_contract,PRECISION

def frozen_source_matrix():
    frozen=read(Path(SOURCE_ROOT)/'frozen_source_matrix.json')
    if any(frozen.get(k)!=v for k,v in dict(status='ALL_E200_FROZEN',run_id=SOURCE_RUN,commit=SOURCE_COMMIT,target_access=False,selection='fixed_all_arms_no_elimination').items()):
        raise ValueError('All8 E200 source freeze required')
    expected=read(SOURCE_CONTRACT);records=[]
    for rid,(v,s) in sorted(expected_rows().items()):
        source=Path(SOURCE_ROOT)/rid/'source';resolved=read(source/'resolved_config.json')
        if resolved.get('commit')!=SOURCE_COMMIT:raise ValueError('Source code commit mismatch')
        records.append(read_source_record(dict(variant=v,model_seed=s,source_output=str(source)),expected,'cvs_reference_residual_identity'))
    key=lambda r:(r['variant'],r['seed'])
    if sorted(frozen.get('records',[]),key=key)!=sorted(records,key=key):raise ValueError('Frozen records differ from source artifacts')
    return frozen

def validate_payload(c,done,initial,contract,expected,resolved,payload):
    validate_config(c);validate_source(resolved)
    if any(contract.get(k)!=v for k,v in expected.items()) or contract.get('classes')!=CLASSES:raise ValueError('CHECKPOINT_DATA_CONTRACT_MISMATCH')
    exact=dict(status='SCRATCH',scratch_only=True,checkpoint=None,ancestors=[],checkpoint_sources=[],target_access=False,target_contact=False,
        model_seed=c['model_seed'],physical_roles='EXACT_MATCH',selection='fixed_last_epoch')
    if initial!=exact:raise ValueError('CHECKPOINT_PROVENANCE_UNVERIFIED_OR_TARGET_CONTAMINATED')
    if any(done.get(k)!=v for k,v in dict(status='SOURCE_TRAINED',epoch=200,steps=10000,target_access=False,target_evaluated=False,checkpoint=c['source_output']+'/last.pt').items()):raise ValueError('Source checkpoint not complete')
    architecture=dual_contract(c['variant'])
    required=dict(method='cvs_reference_residual_identity',variant=c['variant'],model_seed=c['model_seed'],commit=SOURCE_COMMIT,output_root=c['source_output'],
        response=architecture,response_actual=architecture,response_active=True,classifier_scale=30.,precision=PRECISION,
        source_counts={'L_s':6300,'U_s':56700,'V':27000},steps_per_epoch=50,U_s_use='unused',target_access=False,
        total_parameters=architecture['total_parameters'],trainable_parameters=architecture['total_parameters'])
    if any(resolved.get(k)!=v for k,v in required.items()):raise ValueError('Resolved source architecture or budget differs')
    if resolved.get('backend_flags')!=resolved['numerical_policy'] or done.get('backend_flags')!=resolved['numerical_policy']:raise ValueError('Frozen precision differs')
    required_payload=dict(method='cvs_reference_residual_identity',variant=c['variant'],epoch=200,num_classes=6,classes=CLASSES,source_contract=contract,
        initialization=initial,selection='fixed_last_epoch',config=resolved)
    if any(payload.get(k)!=v for k,v in required_payload.items()):raise ValueError('Checkpoint payload disagrees with source artifacts')
    state=payload.get('model')
    if not isinstance(state,dict) or not state or any(not isinstance(t,torch.Tensor) or t.is_complex() or
        (t.is_floating_point() and (t.dtype!=torch.float32 or not torch.isfinite(t).all())) for t in state.values()):raise ValueError('Checkpoint must retain finite FP32')
    probe=build(c['variant']);reference=probe.state_dict()
    if state.keys()!=reference.keys() or any(state[k].shape!=v.shape or state[k].dtype!=v.dtype for k,v in reference.items()):raise ValueError('Frozen tensor schema differs')
    probe.load_state_dict(state,strict=True)
    return probe
