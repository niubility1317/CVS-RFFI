"""Synthetic four-phase export, with frozen native and ABI-safe boundaries."""
import copy
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'code'),str(ROOT/'tools')]
import export_d92_bnna_features as mod
from test_export_d92_mv_kme_features import fixture as mv_fixture,tiny_encoder,dump,CLASSES


def fixture(tmp_path,monkeypatch,k=1):
    args,iq,infer=mv_fixture(tmp_path,monkeypatch,k)
    args['output']=tmp_path/'bnna_features';args['config']=dict(algorithm=copy.deepcopy(mod.local_core().FROZEN_CONFIG))
    provenance=mod.read(args['row_root']/'received_features/checkpoint_provenance.json')
    monkeypatch.setattr(mod,'load_native',lambda **_:(infer,copy.deepcopy(provenance)))
    return args,iq,infer


def test_four_phase_export_is_bound_and_single_observation_invariant(tmp_path,monkeypatch):
    args,iq,infer=fixture(tmp_path,monkeypatch)
    original=Path.open
    def guarded(path,*a,**kw):
        assert path.name not in ('truth.json','scores.json','source_l_features.npz','final_ssdg.pth')
        return original(path,*a,**kw)
    monkeypatch.setattr(Path,'open',guarded)
    def no_numpy_bridge(*_a,**_kw):raise RuntimeError('Legacy/new NumPy C-extension mismatch')
    monkeypatch.setattr(torch.Tensor,'numpy',no_numpy_bridge)
    monkeypatch.setattr(torch,'as_tensor',no_numpy_bridge)
    with threadpool_limits(limits=1):
        mod.export(**args);z,f=mod.transform_received(iq,infer,mod.local_core())
        each=[mod.transform_received(row[None],infer,mod.local_core()) for row in iq]
        rz,rf=mod.transform_received(iq[::-1],infer,mod.local_core())
    np.testing.assert_array_equal(z,np.concatenate([v[0] for v in each]))
    np.testing.assert_array_equal(f,np.concatenate([v[1] for v in each]))
    np.testing.assert_array_equal(z,rz[::-1]);np.testing.assert_array_equal(f,rf[::-1])
    with np.load(args['output']/mod.CACHE_NAME) as data:
        np.testing.assert_array_equal(data['identity_views'],z);np.testing.assert_array_equal(data['fft'],f)
    marker=mod.read(args['output']/'features_complete.json')
    assert marker['status']=='BNNA_FEATURES_COMPLETE' and marker['view_count_per_observation']==4
    assert marker['identity_views_shape']==[len(iq),4,160] and marker['fft_shape']==[len(iq),96]
    assert marker['feature_array_bytes']==len(iq)*(4*160+96)*4
    assert marker['new_ground_statistics_bytes']==0 and marker['model_file_bytes']==123456
    infer.verify_frozen()
    with pytest.raises(FileExistsError):mod.export(**args)


@pytest.mark.parametrize('fault',['sha','capsule','config','origin','ids','shape','nonfinite'])
def test_invalid_binding_rejected_before_output(tmp_path,monkeypatch,fault):
    args,iq,_=fixture(tmp_path,monkeypatch)
    if fault=='sha':args['expected_checkpoint_sha256']='b'*64
    elif fault=='capsule':args['expected_capsule_id']='different'
    elif fault=='config':args['config']['algorithm']['unexpected']=True
    elif fault=='origin':
        path=args['row_root']/'received_features/checkpoint_provenance.json';value=mod.read(path)
        value['checkpoint_inheritance']=['unverified'];dump(path,value)
    else:
        ids=np.asarray([f'physical-{i}' for i in range(len(iq))])
        if fault=='ids':ids[0]=ids[1]
        elif fault=='shape':iq=iq.transpose(0,2,1)
        else:iq[0,0,0]=np.nan
        np.savez(args['capsule']/'received.npz',iq=iq,ids=ids)
    with pytest.raises(ValueError):mod.export(**args)
    assert not args['output'].exists()


def test_exact_loader_uses_full_synthetic_pipeline(tmp_path,monkeypatch,capsys):
    source=tmp_path/'source';source.mkdir();checkpoint=source/'final_ssdg.pth';checkpoint.write_bytes(b'synthetic')
    contract=tmp_path/'contract.json';dump(contract,{})
    calls=[]
    def verify(source,reference,seed):
        calls.append('verify');return dict(classes=CLASSES,num_classes=2),dict(seed=seed)
    monkeypatch.setattr(mod,'verify_source',verify)
    import cvsrffi.checkpoint_loading as loader
    import cvsrffi.identity_only_forward as forward
    monkeypatch.setattr(torch,'load',lambda *_a,**_k:dict(epoch=200,args=dict(seed=123,baseline_ckpt='',teacher_ckpt='',from_scratch=True)))
    tiny=tiny_encoder()
    monkeypatch.setattr(loader,'build_exact_ssdg_model_from_checkpoint',lambda *_a,**_k:(tiny.model,dict(checkpoint_load_strict=True)))
    def infer(model,x,kind):
        assert kind=='z_id' and not torch.is_grad_enabled()
        calls.append((tuple(x.shape),float(x.abs().max())));return model(x),None
    monkeypatch.setattr(forward,'identity_only_feature_forward',infer)
    with threadpool_limits(limits=1):
        instance,provenance=mod.load_native(native_code=ROOT/'code',source=source,contract=contract,seed=123,
            expected_checkpoint_sha256=mod.sha256(checkpoint),device='cpu')
    assert calls[0]=='verify' and len(calls)==9 and all(call[0]==(4,2,256) and call[1]>0 for call in calls[1:])
    smoke=json.loads(capsys.readouterr().out)
    assert smoke['status']=='PASS' and smoke['identity_shape']==[8,4,160] and smoke['fft_shape']==[8,96]
    assert smoke['physical_support_count']==6 and smoke['fold_count']==3 and smoke['singleton_exact']
    assert provenance['query_read_for_smoke'] is False
    instance.verify_frozen()
