import copy
import numpy as np
import pytest
import torch
from experiments.cvs_mirror_mechanism.analyze import packet_metrics,rows_from_arrays,write_csv,analyze
from experiments.cvs_mirror_mechanism.diagnose import diagnose,validate_payload,load_verified_state,validate_source_contract,SOURCE_COMMIT
from experiments.cvs_mirror_subspace_identity.model import build


def test_relative_metric_exposes_near_zero_denominator_without_large_absolute_change():
    a=np.zeros((30,2,31,4,4));b=a.copy()
    a[:,0,0,0,0]=1e-10;b[:,0,0,0,0]=1e-7
    m=packet_metrics(a,b)
    assert np.allclose(m['relative'],999) and np.all(m['absolute']<1e-6)
    assert np.allclose(m['absolute']**2,np.square(m['pair_absolute']).sum(1))


def test_zero_reference_has_finite_metric_and_explicit_absolute_change():
    a=np.zeros((30,2,31,4,4));m=packet_metrics(a,a)
    assert all(np.all(v==0) for v in m.values())


@pytest.mark.parametrize('bad',['shape','nan'])
def test_invalid_population_rejected(bad):
    a=np.zeros((30,2,31,4,4));b=a.copy()
    if bad=='shape':b=b[:18]
    else:b[0,0,0,0,0]=np.nan
    with pytest.raises(ValueError):packet_metrics(a,b)


@pytest.mark.parametrize('variant',['mirror_energy','mirror_subspace'])
def test_full_frozen_capture_keeps_all_groups_and_state(variant):
    torch.set_num_threads(2);torch.manual_seed(2026100301)
    model=build(variant);old={k:v.clone() for k,v in model.state_dict().items()}
    arrays=diagnose(model)
    packets,pairs=rows_from_arrays(arrays,variant,2026092701)
    assert len(packets)==180 and len(pairs)==5580
    assert {r['signal_group'] for r in packets}=={'noise','tone','periodic','dc','zero'}
    assert all(torch.equal(old[k],v) for k,v in model.state_dict().items())
    assert all(p.grad is None for p in model.parameters())
    assert np.all(arrays['fp32_base_q'][24:]==0)
    assert np.all(arrays['fp64_base_q'][24:]==0)
    assert arrays['fp32_base_q'].dtype==np.float32 and arrays['fp64_base_q'].dtype==np.float64


def payload_fixture():
    init=dict(status='SCRATCH',ancestors=[],checkpoint_sources=[],physical_roles='EXACT_MATCH',target_access=False,target_contact=False)
    row=dict(resolved=dict(commit=SOURCE_COMMIT,variant='mirror_subspace'),initialization=init)
    contract=dict(classes=list('abcdef'),role_ids={'L_s':list(range(6300)),'U_s':list(range(56700)),'V':list(range(27000))})
    payload=dict(method='cvs_mirror_subspace_identity',variant='mirror_subspace',epoch=200,num_classes=6,
        initialization=copy.deepcopy(init),selection='fixed_last_epoch',config=copy.deepcopy(row['resolved']),
        classes=list('abcdef'),source_contract=contract)
    return row,payload


def test_checkpoint_must_match_actual_audited_source_metadata():
    row,payload=payload_fixture();assert validate_payload(payload,row)==payload['source_contract']
    payload['config']['commit']='other'
    with pytest.raises(ValueError):validate_payload(payload,row)


@pytest.mark.parametrize('mutation',['target','ancestors','physical_roles','classes','epoch'])
def test_tainted_or_mismatched_checkpoint_rejected(mutation):
    row,payload=payload_fixture()
    if mutation=='target':row['initialization']['target_access']=True
    elif mutation=='ancestors':row['initialization']['ancestors']=['old']
    elif mutation=='physical_roles':payload['source_contract']['role_ids']['V'].pop()
    elif mutation=='classes':payload['classes']=['wrong']*6
    else:payload['epoch']=199
    with pytest.raises(ValueError):validate_payload(payload,row)


@pytest.mark.parametrize('mutation',['fp64','window','nan','contract'])
def test_raw_checkpoint_state_checked_before_torch_can_cast(mutation):
    model=build('mirror_subspace');state={k:v.clone() for k,v in model.state_dict().items()};contract=model.contract()
    if mutation=='fp64':state={k:v.double() if v.is_floating_point() else v for k,v in state.items()}
    elif mutation=='window':state['core.mirror_relation.window'].fill_(1)
    elif mutation=='nan':state['core.mirror_relation.mix_real'][0,0]=float('nan')
    else:contract['mirror_relation_active']=False
    with pytest.raises(ValueError):load_verified_state(model,state,contract)


def test_real_fp32_state_and_architecture_contract_load():
    model=build('mirror_energy');load_verified_state(model,model.state_dict(),model.contract())


def test_existing_analysis_is_never_replaced(tmp_path):
    e=tmp_path/'report/evidence';e.mkdir(parents=True);p=e/'public_packets.csv';p.write_text('preserve',encoding='utf-8')
    with pytest.raises(FileExistsError):write_csv(p,[dict(x=1)])
    with pytest.raises(FileExistsError):analyze(tmp_path/'unused',tmp_path/'report')
    assert p.read_text(encoding='utf-8')=='preserve'


@pytest.mark.parametrize('mutation',[None,'reference','extension','unknown'])
def test_original_contract_with_exact_training_extensions(mutation):
    row=dict(resolved=dict(dataset='/source/ManySig.pkl'))
    reference=dict(role_ids={'V':['known_id']},source_rxs=[1,3,4,6,8])
    actual=copy.deepcopy(reference)
    actual.update(equalized=1,out_len=256,classes=['14-10','14-7','20-15','20-19','6-15','8-20'],
                  physical_roles='EXACT_MATCH',dataset_path='/source/ManySig.pkl',normalize=True)
    if mutation=='reference':actual['role_ids']['V']=['wrong_id']
    elif mutation=='extension':actual['equalized']=0
    elif mutation=='unknown':actual['target_access']=True
    if mutation is None:validate_source_contract(actual,reference,row)
    else:
        with pytest.raises(ValueError):validate_source_contract(actual,reference,row)
