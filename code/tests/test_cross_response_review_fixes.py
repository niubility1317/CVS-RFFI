import sys
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from cvsrffi.cross_response.analysis import paired_joint_effects_by_variant,SCENARIOS
from cvsrffi.cross_response.integration import input_artifact_identity,CrossResponseRuntime
from scripts.core90_cross_response_matrix import load_config,DEFAULT_CONFIG


def test_actual_matrix_names_produce_separate_joint_effects():
    rows = [dict(variant=v,seed=s,metrics={scene:{'accuracy':accuracy} for scene in SCENARIOS})
            for s in (1,2) for v,accuracy in (('U1',.70),('U2',.75),('U3',.76),
                                              ('U4_additive',.83),('U4_bilinear',.86))]
    reports = paired_joint_effects_by_variant(rows,bootstrap_samples=100)
    assert reports['U4_additive']['scenarios']['clean']['mean']==pytest.approx(.02)
    assert reports['U4_bilinear']['scenarios']['clean']['mean']==pytest.approx(.05)
    assert all(r['scenarios']['clean']['paired_seed_count']==2 for r in reports.values())
    assert reports['U4_additive']['matched_factorial']
    assert reports['U4_additive']['scenarios']['clean']['confidence_interval'] is not None
    assert reports['U4_bilinear']['capacity_confounded']
    assert not reports['U4_bilinear']['matched_factorial']
    assert reports['U4_bilinear']['scenarios']['clean']['confidence_interval'] is None
    assert reports['U4_bilinear']['scenarios']['clean']['interpretation']=='descriptive_only'


def test_resume_binds_identical_local_record_ids_to_actual_dataset(tmp_path):
    original,other = tmp_path/'source.pkl',tmp_path/'other.pkl'
    original.write_bytes(b'original IQ bytes')
    other.write_bytes(b'different IQ bytes')
    baseline = load_config()['baseline_args']
    args = SimpleNamespace(**baseline)
    args.cross_response_config = str(DEFAULT_CONFIG)
    args.wisig_pkl = str(original)
    contract = {'input_artifact':input_artifact_identity(original),
                'train':((0,0,0,0,0),),'validation':((0,0,0,0,1),)}
    runtime = SimpleNamespace(config={},variant='U5',source_contract=contract)
    payload = dict(cross_response=dict(initialization='scratch',config={},variant='U5',source_contract=contract),
                   checkpoint_selection='final_only',checkpoint_role='cross_response_epoch_resume',args=vars(args).copy())
    CrossResponseRuntime.validate_checkpoint(runtime,payload,args)
    # Local physical indices/roles stay exactly identical; only the real input changes.
    runtime.source_contract = dict(contract,input_artifact=input_artifact_identity(other))
    args.wisig_pkl = str(other)
    with pytest.raises(ValueError,match='CHECKPOINT_DATA_CONTRACT_MISMATCH'):
        CrossResponseRuntime.validate_checkpoint(runtime,payload,args)
    # Replacing bytes at the same path also invalidates inherited training state.
    original.write_bytes(b'replaced IQ bytes')
    runtime.source_contract = dict(contract,input_artifact=input_artifact_identity(original))
    args.wisig_pkl = str(original)
    with pytest.raises(ValueError,match='CHECKPOINT_DATA_CONTRACT_MISMATCH'):
        CrossResponseRuntime.validate_checkpoint(runtime,payload,args)


def _metadata_runtime(tmp_path,unlabeled_rows=None,validation_rows=None):
    def dataset(split,rows):
        # No __getitem__: verification must remain metadata-only.
        return SimpleNamespace(split_source=split,index=[SimpleNamespace(tx_i=t,rx_i=r,day_i=d,eq_i=0,sig_i=s)
                               for t,r,d,s in rows])
    source = tmp_path/'source.pkl'
    if not source.exists(): source.write_bytes(b'unchanged dataset artifact')
    args = SimpleNamespace(**load_config()['baseline_args'])
    args.cross_response_variant = 'U0'
    args.cross_response_config = str(DEFAULT_CONFIG)
    args.wisig_pkl,args.output_dir = str(source),str(tmp_path/'out')
    args.wisig_train_rxs,args.wisig_test_rxs = '0','1'
    args.wisig_train_days,args.wisig_test_days = '0','1'
    ctx = {'train_loader':SimpleNamespace(dataset=dataset('ssdg_labeled_tx_visible',[(0,0,0,0)])),
           'val_loader':SimpleNamespace(dataset=dataset('ssdg_source_v_select',validation_rows or [(0,0,0,1)]))}
    if unlabeled_rows is not None:
        ctx['unlabeled_loader'] = SimpleNamespace(dataset=dataset('ssdg_unlabeled_tx_hidden',unlabeled_rows))
    runtime = CrossResponseRuntime(None,ctx,load_config()['cross_response'],args,torch.device('cpu'))
    return runtime,args


def test_unlabeled_contract_is_optional_but_bound_and_role_disjoint(tmp_path):
    absent,_ = _metadata_runtime(tmp_path)
    assert absent.source_contract['unlabeled']==()
    original,args = _metadata_runtime(tmp_path,[(0,0,0,2)])
    assert original.source_contract['split_roles']['unlabeled']=='ssdg_unlabeled_tx_hidden'
    payload = dict(cross_response=original.state_dict(),checkpoint_selection='final_only',
                   checkpoint_role='cross_response_epoch_resume',args=vars(args).copy())
    changed,args2 = _metadata_runtime(tmp_path,[(0,0,0,3)])
    assert original.source_contract['input_artifact']==changed.source_contract['input_artifact']
    with pytest.raises(ValueError,match='CHECKPOINT_DATA_CONTRACT_MISMATCH'):
        changed.validate_checkpoint(payload,args2)
    for duplicate in (0,1):
        with pytest.raises(ValueError,match='physical records overlap'):
            _metadata_runtime(tmp_path,[(0,0,0,duplicate)])


@pytest.mark.parametrize('rows', [[(0,1,0,2)],[(0,0,1,2)]])
def test_actual_unlabeled_source_receiver_and_day_are_checked(tmp_path,rows):
    with pytest.raises(ValueError,match='outside declared source RX/day'):
        _metadata_runtime(tmp_path,rows)
