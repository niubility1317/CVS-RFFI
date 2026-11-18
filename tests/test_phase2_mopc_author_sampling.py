import torch
import pytest

from experiments.pamf_phase2_practical.mopc_author import MoPCAuthorWiSig256
from experiments.pamf_phase2_practical.vendor_native.comparison_suite import registration as cil


def test_author_drops_incomplete_training_batch_and_extracts_from_full_batches(monkeypatch):
    torch.set_num_threads(1)
    model = MoPCAuthorWiSig256().eval()
    anchor = cil.RegistrationState('mopc_hr', model, tuple(range(6)), 6,
                                   prototypes=torch.zeros(6, 256))
    x = torch.randn(25, 2, 256)
    y = torch.arange(6, 11).repeat_interleave(5)
    calls=[]
    original=cil._features
    def observe(model, method, data, batch_size):
        calls.append((len(data),batch_size))
        return original(model,method,data,batch_size)
    monkeypatch.setattr(cil,'_features',observe)
    result=cil.register_new_classes(anchor,x,y,tuple(range(6,11)),
        config={'seed':4,'mopc_epochs':1,'mopc_batch_size':16,'mopc_stage_size':5})
    assert len(result.loss_trace)==1
    assert calls[-2:]==[(16,16),(16,16)]
    assert result.protocol_metadata['mopc_sampling_version']=='author_drop_last_v1'
    assert result.resource['mopc_sampling_version']=='author_drop_last_v1'


def test_author_reports_zero_batch_without_training():
    model=MoPCAuthorWiSig256().eval()
    anchor=cil.RegistrationState('mopc_hr',model,tuple(range(6)),6,prototypes=torch.zeros(6,256))
    x=torch.randn(10,2,256);y=torch.arange(6,11).repeat_interleave(2)
    with pytest.raises(cil.MoPCSamplingError, match='no complete support batch'):
        cil.register_new_classes(anchor,x,y,tuple(range(6,11)),
            config={'seed':4,'mopc_epochs':1,'mopc_batch_size':16,'mopc_stage_size':5})


def test_author_reports_missing_class_in_prototype_pass():
    torch.set_num_threads(1)
    model=MoPCAuthorWiSig256().eval()
    anchor=cil.RegistrationState('mopc_hr',model,tuple(range(6)),6,prototypes=torch.zeros(6,256))
    x=torch.randn(25,2,256);y=torch.arange(6,11).repeat_interleave(5)
    with pytest.raises(cil.MoPCSamplingError, match='omitted a registered class'):
        cil.register_new_classes(anchor,x,y,tuple(range(6,11)),
            config={'seed':6,'mopc_epochs':1,'mopc_batch_size':16,'mopc_stage_size':5})


def test_satellite_worker_keeps_sampling_failure_distinct(tmp_path, monkeypatch):
    from experiments.pamf_phase2_practical import satellite_mix3,group2_worker
    out=tmp_path/'out';out.mkdir()
    monkeypatch.setattr(satellite_mix3,'read',lambda path:{'input_condition':'satellite_received'})
    def fail(rt,job):
        raise cil.MoPCSamplingError('prototype class omitted')
    monkeypatch.setattr(group2_worker,'predict',fail)
    job=dict(group='hybrid',profile='protonet_cda',methods=['mopc_hr'],capsule='cap',seed=2,output=str(out))
    with pytest.raises(cil.MoPCSamplingError):
        satellite_mix3.run_worker(dict(runtimes={'hybrid':'unused'}),job)
    import json
    marker=json.loads((out/'numerical_failure.json').read_text())
    assert marker['status']=='MOPC_SAMPLING_FAILURE'
    assert marker['query_fit'] is False and marker['truth_read'] is False


def test_scoring_rejects_older_public_code_sampling_when_new_version_is_required():
    from experiments.pamf_phase2_practical.mopc_identity import require_author_sampling
    row={'resources':{'C_protocol':{'mopc_version':'public_code_semantics_matched_IQ_extension'}}}
    with pytest.raises(ValueError, match='sampling'):
        require_author_sampling(row)
    row['resources']['C_protocol']['mopc_sampling_version']='author_drop_last_v1'
    require_author_sampling(row)


def test_source_prototypes_use_seeded_shuffled_complete_batches(monkeypatch):
    from experiments.pamf_phase2_practical import group1_source
    class Model:
        training=False
        out_dim=2
    x=torch.arange(60,dtype=torch.float32).reshape(30,2)
    y=torch.arange(6).repeat_interleave(5)
    seen=[]
    def extract(model,method,values,device,batch=128):
        seen.append(values.clone())
        assert batch==16
        return values
    monkeypatch.setattr(group1_source,'extract',extract)
    metadata=group1_source.mopc_source_metadata(Model(),x,y,'cpu',seed=0)
    assert len(seen[0])==16
    assert metadata['source_rows']==30
    assert metadata['source_rows_used']==16
    assert metadata['source_sampling']=='author_shuffled_drop_last'
