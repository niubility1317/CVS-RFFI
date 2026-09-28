"""Received-only export using a tiny frozen synthetic encoder, never a GPU."""
import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'code'),str(ROOT/'tools')]
import export_d92_mv_kme_features as mod

SHA='a'*64
CLASSES=['old-z','old-a']


def dump(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value),encoding='utf-8')


def tiny_encoder():
    torch.manual_seed(912)
    model=torch.nn.Sequential(torch.nn.Flatten(),torch.nn.Linear(512,160),torch.nn.BatchNorm1d(160))
    return mod.FrozenIdentity(model,lambda m,x,_:(m(x),None),torch.device('cpu'))


def fixture(tmp_path,monkeypatch,k=1):
    capsule=tmp_path/'capsule';capsule.mkdir()
    count=4*k+3;ids=np.asarray([f'physical-{index}' for index in range(count)])
    iq=np.random.default_rng(12).normal(size=(count,2,256)).astype(np.float32)
    np.savez(capsule/'received.npz',iq=iq,ids=ids)
    dump(capsule/'manifest.json',dict(protocol_schema='p2_min_v1',phase2_data_status='VALIDATED_ONCE',
        capsule_id='synthetic',split_count=1))
    dump(capsule/'splits/one.json',dict(protocol_schema='p2_min_v1',phase2_data_status='VALIDATED_ONCE',
        capsule_id='synthetic',split_id='one',registered_classes=[*CLASSES,'new-z','new-a'],
        support_indices=list(range(4*k)),support_labels=np.repeat(np.arange(4),k).tolist(),
        query_indices=list(range(4*k,count)),k=k,support_seed=1,receiver='rx',scenario='synthetic'))
    row=tmp_path/'old_row'
    dump(row/'d92_startup.json',dict(checkpoint_sha256=SHA,capsule=str(capsule),seed=123,
        features=str(row/'received_features/received_features.npz'),truth_read=False,query_fit_access=False))
    provenance=dict(verdict='MATCHED_SOURCE_ONLY_SCRATCH',checkpoint_sha256=SHA,checkpoint_epoch=200,
        checkpoint_inheritance=[],target_access_before_freeze=False,source_role_comparison='EXACT_MATCH',
        classes=CLASSES,model_seed=123,model_file_bytes=123456)
    dump(row/'received_features/checkpoint_provenance.json',provenance)
    infer=tiny_encoder()
    monkeypatch.setattr(mod,'load_native',lambda **_: (infer,copy.deepcopy(provenance)))
    return dict(row_root=row,source=tmp_path/'unopened_source',contract=tmp_path/'unopened_contract',
        native_code=tmp_path/'unopened_native',seed=123,capsule=capsule,output=tmp_path/'mv_features',
        config=dict(algorithm=copy.deepcopy(mod.local_core().FROZEN_CONFIG)),
        expected_capsule_id='synthetic',expected_checkpoint_sha256=SHA,device='cpu'),iq,infer


def test_export_bound_cache_and_query_independent_features(tmp_path,monkeypatch):
    args,iq,infer=fixture(tmp_path,monkeypatch)
    original=Path.open;opened=[]
    def guarded(path,*a,**kw):
        assert path.name not in ('truth.json','scores.json','source_l_features.npz','final_ssdg.pth')
        opened.append(path);return original(path,*a,**kw)
    monkeypatch.setattr(Path,'open',guarded)
    with threadpool_limits(limits=1):
        mod.export(**args)
        batch=mod.transform_received(iq,infer,mod.local_core())
        singleton=np.concatenate([mod.transform_received(iq[i:i+1],infer,mod.local_core()) for i in range(len(iq))])
        reordered=mod.transform_received(iq[::-1],infer,mod.local_core())[::-1]
    np.testing.assert_array_equal(batch,singleton)
    np.testing.assert_array_equal(batch,reordered)
    with np.load(args['output']/'received_mv_features.npz') as data:
        np.testing.assert_array_equal(data['blocks'],batch)
        assert data['blocks'].dtype==np.float32 and data['blocks'].shape==(len(iq),3,256)
        assert data['checkpoint_sha256'].item()==SHA
    marker=mod.read(args['output']/'features_complete.json')
    assert marker['status']=='MULTIVIEW_FEATURES_COMPLETE' and marker['new_ground_statistics_bytes']==0
    assert marker['model_file_bytes']==123456
    assert marker['feature_array_bytes']==len(iq)*3*256*4
    infer.verify_frozen()
    assert opened
    with pytest.raises(FileExistsError):mod.export(**args)


@pytest.mark.parametrize('fault',['config','sha','capsule','origin','ids','shape','nonfinite'])
def test_invalid_received_binding_rejected(tmp_path,monkeypatch,fault):
    args,iq,_=fixture(tmp_path,monkeypatch)
    if fault=='config':args['config']['algorithm']['fourier_seed']=42
    elif fault=='sha':args['expected_checkpoint_sha256']='b'*64
    elif fault=='capsule':args['expected_capsule_id']='other'
    elif fault=='origin':
        path=args['row_root']/'received_features/checkpoint_provenance.json'
        provenance=mod.read(path);provenance['checkpoint_inheritance']=['ancestor'];dump(path,provenance)
    else:
        ids=np.asarray([f'physical-{index}' for index in range(len(iq))])
        if fault=='ids':ids[1]=ids[0]
        elif fault=='shape':iq=iq.transpose(0,2,1)
        else:iq[0,0,0]=np.nan
        np.savez(args['capsule']/'received.npz',iq=iq,ids=ids)
    with pytest.raises(ValueError):mod.export(**args)
    assert not args['output'].exists()


@pytest.mark.parametrize('fault',['training','requires_grad','buffer','parameter'])
def test_native_frozen_state_detects_mutation(fault):
    infer=tiny_encoder()
    if fault=='training':infer.model.train()
    elif fault=='requires_grad':next(infer.model.parameters()).requires_grad_(True)
    elif fault=='buffer':next(infer.model.buffers()).add_(1)
    else:
        with torch.no_grad():next(infer.model.parameters()).add_(1)
    with pytest.raises(ValueError):infer.verify_frozen()


def test_exact_native_loader_metadata_hash_and_synthetic_smoke(tmp_path,monkeypatch,capsys):
    source=tmp_path/'source';source.mkdir()
    checkpoint=source/'final_ssdg.pth';checkpoint.write_bytes(b'synthetic checkpoint fixture')
    contract=tmp_path/'contract.json';dump(contract,{})
    calls=[]
    def verify(actual_source,reference,seed):
        calls.append(('verify',actual_source,seed));return dict(classes=CLASSES,num_classes=2),dict(seed=seed)
    monkeypatch.setattr(mod,'verify_source',verify)
    import cvsrffi.checkpoint_loading as loader
    import cvsrffi.identity_only_forward as forward
    monkeypatch.setattr(torch,'load',lambda *_a,**_k:dict(epoch=200,args=dict(seed=123,baseline_ckpt='',teacher_ckpt='',from_scratch=True)))
    infer=tiny_encoder()
    monkeypatch.setattr(loader,'build_exact_ssdg_model_from_checkpoint',lambda *_a,**_k:(infer.model,dict(checkpoint_load_strict=True)))
    def native_forward(model,x,kind):
        assert not torch.is_grad_enabled() and kind=='z_id'
        calls.append(('forward',tuple(x.shape),float(x.abs().max())))
        return model(x),None
    monkeypatch.setattr(forward,'identity_only_feature_forward',native_forward)
    result,provenance=mod.load_native(native_code=ROOT/'code',source=source,contract=contract,seed=123,
        expected_checkpoint_sha256=mod.sha256(checkpoint),device='cpu')
    assert calls[0][0]=='verify' and calls[1]==('forward',(16,2,256),0.0)
    assert provenance['model_file_bytes']==checkpoint.stat().st_size
    assert provenance['query_read_for_smoke'] is False
    assert json.loads(capsys.readouterr().out)['status']=='PASS'
    result.verify_frozen()


def test_feature_progress_every_128_observations_and_final(capsys):
    core=SimpleNamespace(make_received_views=lambda _:np.zeros((1,4,4,2,256)),
        build_fourier_blocks=lambda **_:np.zeros((1,3,256)))
    mod.transform_received(np.zeros((129,2,256)),lambda _:np.zeros((16,160)),core)
    events=[json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert [e['completed'] for e in events]==[128,129]
    assert [e['views'] for e in events]==[128*16,129*16]
    assert all(e['total']==129 and e['event']=='FEATURE_PROGRESS' and e['extraction_elapsed_seconds']>=0 for e in events)
    assert all(not isinstance(value,(list,dict)) for e in events for value in e.values())
