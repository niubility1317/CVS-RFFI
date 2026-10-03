import copy,json
from pathlib import Path
import pytest
from experiments.cvs_reference_residual_identity.model import VARIANTS
from experiments.cvs_reference_residual_identity.contracts import PRECISION
from experiments.cvs_reference_residual_identity.source import validate_config
from experiments.cvs_reference_residual_identity import dispatch as d
ROOT=Path(__file__).resolve().parents[1]

def config(v=VARIANTS[0],seed=2026092701):
    return json.loads((ROOT/'experiments/cvs_reference_residual_identity/source_configs'/f'{v}-s{seed}.json').read_text(encoding='utf-8'))

def test_all_eight_source_configs_reject_changed_data_training_and_inheritance():
    for v in VARIANTS:
        for seed in sorted(d.SEEDS):
            c=config(v,seed);validate_config(c)
            for key,value in [('target_truth','truth'),('target_capsule','target'),('checkpoint','old'),('teacher','old'),('ancestors',['old']),
                              ('epochs',201),('extra_losses',['consistency']),('augmentation',True),('dataset','other'),('source_contract','other'),
                              ('response',{}),('model_seed',7),('numerical_policy',{})]:
                bad=copy.deepcopy(c);bad[key]=value
                with pytest.raises(ValueError):validate_config(bad)

def test_fixed_all_eight_matrix_rejects_duplicates_and_target(tmp_path,monkeypatch):
    monkeypatch.setattr(d,'PROJECT',str(tmp_path))
    spec=dict(run_id=d.RUN,runtime_root=str(tmp_path/'runs'/d.RUN),log_root=str(tmp_path/'logs'/d.RUN),rows=[])
    for v in VARIANTS:
        for seed in sorted(d.SEEDS):
            c=config(v,seed);rid=f'{v}-s{seed}';c['output_root']=str(tmp_path/'runs'/d.RUN/rid/'source')
            p=tmp_path/(rid+'.json');p.write_text(json.dumps(c),encoding='utf-8')
            spec['numerical_policy']=c['numerical_policy'];spec['rows'].append(dict(row_id=rid,variant=v,model_seed=seed,source_output=c['output_root'],source_config=str(p)))
    d.validate_spec(spec)
    for change in ('duplicate','missing','target','outside'):
        bad=copy.deepcopy(spec)
        if change=='duplicate':bad['rows'][-1]=bad['rows'][0]
        elif change=='missing':bad['rows'].pop()
        elif change=='target':bad['target_inputs']='forbidden'
        else:bad['rows'][0]['source_output']=str(tmp_path/'other')
        with pytest.raises(ValueError):d.validate_spec(bad)

@pytest.mark.parametrize('corruption',['roles','ancestor','target','budget','precision','actual','parameters','inactive','flags'])
def test_freeze_requires_complete_actual_source_and_scratch(tmp_path,corruption):
    c=config();c['output_root']=str(tmp_path)
    original=dict(role_ids=dict(L_s=['L'],U_s=['U'],V=['V']),counts=dict(L_s=6300,U_s=56700,V=27000))
    contract=dict(copy.deepcopy(original),physical_roles='EXACT_MATCH',dataset_path=c['dataset'],classes=['14-10','14-7','20-15','20-19','6-15','8-20'],equalized=1,out_len=256,normalize=True)
    initial=dict(status='SCRATCH',scratch_only=True,checkpoint=None,ancestors=[],checkpoint_sources=[],target_access=False,target_contact=False,model_seed=c['model_seed'],physical_roles='EXACT_MATCH',selection='fixed_last_epoch')
    resolved=dict(c,steps_per_epoch=50,source_counts=original['counts'],U_s_use='unused',target_access=False,precision=PRECISION,
        gradient_clipping=None,optimizer='AdamW+CosineAnnealingLR',loader_seed=c['model_seed'],backend_flags=c['numerical_policy'],
        response_actual=copy.deepcopy(c['response']),response_active=True,classifier_scale=30.,total_parameters=189562)
    done=dict(status='SOURCE_TRAINED',epoch=200,steps=10000,target_access=False,target_evaluated=False,backend_flags=c['numerical_policy'],
        final_source_metrics=dict(source_val_count=27000,source_val_accuracy=.9,source_val_worst_rx=.9,source_val_rx_accuracy={str(rx):.9 for rx in [1,3,4,6,8]}))
    profile=dict(total_parameters=189562,gradient_used_parameters=189562,conv_linear_macs_per_sample=1)
    files={'completion.json':done,'resolved_config.json':resolved,'initialization.json':initial,'source_contract.json':contract,'resource_profile.json':profile}
    def save():
        for name,value in files.items():(tmp_path/name).write_text(json.dumps(value),encoding='utf-8')
    row=dict(variant=c['variant'],model_seed=c['model_seed'],source_output=str(tmp_path))
    save();assert d.read_source_record(row,original,c['method'])['parameters']==189562
    if corruption=='roles':contract['role_ids']['V']=['other']
    elif corruption=='ancestor':initial['ancestors']=['old.pt']
    elif corruption=='target':done['target_evaluated']=True
    elif corruption=='budget':done['steps']=9999
    elif corruption=='precision':resolved['precision']='float32'
    elif corruption=='actual':resolved['response_actual']['window']=[0,80]
    elif corruption=='parameters':profile['total_parameters']=1
    elif corruption=='inactive':resolved['response_active']=False
    elif corruption=='flags':done['backend_flags']={}
    save()
    with pytest.raises(ValueError):d.read_source_record(row,original,c['method'])
