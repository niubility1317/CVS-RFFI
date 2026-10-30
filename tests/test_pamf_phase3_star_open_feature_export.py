import copy
import json
from pathlib import Path
import numpy as np
import pytest
import torch
from torch import nn
from experiments.pamf_phase3_open_world import star_open_feature_export as e
from tests.test_pamf_phase3_star_open_world_train import config as parent_config


def config(output):
    parents={}
    for arm in e.ARMS:
        t=copy.deepcopy(parent_config());out=str(Path('synthetic-parents')/arm/'train')
        t.update(arm=arm,code_commit=e.PARENT_COMMIT,output_root=out)
        t['native_config'].update(row_id=arm+'-s2026092701',code_commit=e.PARENT_COMMIT,output_root=out)
        parents[arm]=t
    return dict(run_id=e.RUN_ID,code_commit='b'*40,train_configs=parents,cache_root='synthetic-cache',output_root=str(output),batch_size=128)


def test_source_only_config_and_original_parent_identity(tmp_path):
    c=config(tmp_path/'new');assert e.validate_config(c) is c
    for key in ('truth','target_root','Q_public_root','OE_truth','source_V'):
        with pytest.raises(ValueError):e.validate_config(dict(c,**{key:'forbidden'}))
    changed=copy.deepcopy(c);changed['train_configs']['M']['code_commit']='c'*40
    with pytest.raises(ValueError):e.validate_config(changed)
    changed=copy.deepcopy(c);changed['output_root']=changed['cache_root']
    with pytest.raises(ValueError):e.validate_config(changed)


def test_all_smokes_precede_source_cache_and_frozen_complete(monkeypatch,tmp_path):
    c=config(tmp_path/'new');events=[];models={a:nn.Linear(2,2).eval().requires_grad_(False) for a in ('F0','S0','S1','M')}
    for a,m in models.items():m.name=a
    ids=np.asarray(['n'+str(i) for i in range(6)]);meta=dict(rx_i=np.asarray([1,3,4,6,8,1],np.int64),day_i=np.asarray([1,1,2,2,3,3],np.int64),domain=np.arange(6,dtype=np.int64))
    manifest=dict(role='novel',count=6,status='STAR_REPRESENTATION_ROLE_COMPLETE',parent_F0_identity=dict(parent=True),dura_identity=dict(dura=True),all_new_IQ_satellite=True,restored_input_path='P1')
    bundle=dict(manifest=manifest,ids=ids,metadata=meta,received=object(),restored=np.zeros((6,2,256),np.float32))
    def role(path,which):
        assert path==Path('synthetic-cache/novel') and which=='novel'
        assert events==['F0:2','S0:2','S1:2','M:2'];events.append('load_novel');return bundle
    def features(m,x,is_original=False):
        events.append(m.name+':'+str(len(x)));keys={'fused160':160} if is_original else {'fused160':160,'open64':64}
        return {k:np.tile(np.eye(1,d,dtype=np.float32),(len(x),1)) for k,d in keys.items()}
    monkeypatch.setattr(e,'COUNT',6);monkeypatch.setattr(e,'_backend',lambda c,device:(models,{'F0':dict(parent=True)},dict(dura=True)))
    monkeypatch.setattr(e,'_synthetic',lambda device:torch.zeros((2,2,256)));monkeypatch.setattr(e,'_features',features)
    monkeypatch.setattr(e,'load_role',role);monkeypatch.setattr(e,'close_role',lambda value:events.append('close'))
    done=e.export(c,'cpu');assert done['model_states_unchanged'] and done['target_reads']==done['private_truth_reads']==done['optimizer_updates']==0
    assert done['real_F_forward_rows']==24 and done['synthetic_F_forward_rows']==8 and len(done['representations'])==7
    for a in models:
        with np.load(tmp_path/'new'/(a+'.npz'),allow_pickle=False) as z:
            assert np.array_equal(z['ids'],ids) and set(z.files)==({'ids','rx_i','day_i','fused160'}|({'open64'} if a!='F0' else set()))
    with pytest.raises(FileExistsError):e.export(c,'cpu')


def test_same_pass_frozen_fp32_features_and_zero_vectors(monkeypatch):
    m=nn.Linear(2,160).eval().requires_grad_(False);m.features=lambda x:m(x.mean(2));x=torch.randn((3,2,256));calls=[]
    def forward(model,value):
        calls.append(1);z=model(value.mean(2));return dict(raw_fused160=z,open64=z[:,:64])
    monkeypatch.setattr(e.ow,'forward',forward);monkeypatch.setattr(e.ow,'clear',lambda m:None)
    before=e._snapshot_model(m)
    with torch.autocast(device_type='cpu',dtype=torch.bfloat16):values=e._features(m,x)
    assert len(calls)==1 and set(values)=={'fused160','open64'}
    assert all(v.dtype==np.float32 and np.allclose(np.linalg.norm(v,axis=1),1.,atol=2e-6) for v in values.values())
    e._check_model(m,before)
    with torch.no_grad():m.weight.zero_();m.bias.zero_()
    values=e._features(m,x,is_original=True);assert np.count_nonzero(values['fused160'])==0
