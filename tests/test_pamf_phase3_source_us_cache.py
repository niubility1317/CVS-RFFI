"""Synthetic original-ID raw lookup, single observation and TX-blind public Us."""
from copy import deepcopy
import hashlib
import importlib
from pathlib import Path
import numpy as np
import pytest
import torch


def api():return importlib.import_module('experiments.pamf_phase3_open_world.source_us_cache')


def sid(tx,rx,day,eq,sig):return hashlib.sha256(f'{tx}/{rx}/{day}/{eq}/{sig}'.encode()).hexdigest()[:24]


def fixture(root,mp):
    m=api();mp.setattr(m,'COUNT',30);s=m.source
    cache=root/'original';cache.mkdir();mix=root/'mixture.json';mix.write_text(Path('configs/pamf_phase2_practical_mixture_20261028.json').read_text(encoding='utf-8'),encoding='utf-8')
    reader=dict(base_release='base',source_output=str(root/'originalF0'),model_seed=2026092701,frozen_sources={k:'frozen-'+k for k in ('F','AF','BASE','FOLD','G')})
    sc=dict(run_id=m.cache.RUN_ID,code_commit='a'*40,reader_config=reader,source_cache=str(cache),source_contract=str(root/'contract.json'),raw_mainsig=str(root/'ManySig.pkl'),mixture_config_path=str(mix),
        episode=s.EPISODE.copy(),dura_checkpoint=str(root/'G/final.pt'),dura_release=str(root/'G-release'),output_root=str(root/'old_cache'),batch_size=128)
    c=dict(run_id=m.RUN_ID,code_commit='b'*40,source_config=sc,output_root=str(root/'new/cache'),batch_size=128)
    records=[(tx,rx,day,1,1) for rx in s.RECEIVERS for day in (1,2,3) for tx in (0,1)]
    # Preserve an opaque non-sorted contract order, rather than accidental hash sort.
    order=np.random.default_rng(71).permutation(30);records=[records[i] for i in order];ids=np.asarray([sid(*r) for r in records]);meta=dict(ids=ids,**{k:np.asarray([r[j] for r in records],np.int64) for k,j in (('rx_i',1),('day_i',2),('eq_i',3),('sig_i',4))},base_index=np.arange(30,dtype=np.int64),domain=np.full(30,35,np.int64))
    np.savez(cache/'U_s_metadata.npz',**meta)
    contract=dict(classes=s.CLASSES,source_rxs=list(s.RECEIVERS),source_days=[1,2,3],split_seed=392005,counts=dict(L_s=1,U_s=30,V=1),
        role_ids=dict(U_s=ids.tolist(),L_s=[sid(2,1,1,1,0)],V=[sid(3,1,1,1,0)]),num_classes=6)
    m._write(root/'contract.json',contract);events=[]
    def contract_read(*a):events.append('metadata');return deepcopy(contract)
    mp.setattr(m.source,'_contract',contract_read)
    data=[[[[None,None] for _ in range(4)] for _ in s.MAIN_RX_NAMES] for _ in range(6)]
    for tx in range(6):
        for rx in s.RECEIVERS:
            for day in (1,2,3):
                t=np.arange(256);data[tx][rx][day][1]=np.stack([np.stack(((2+tx+.1*sig)*np.cos(.07*t),(2+tx+.1*sig)*np.sin(.07*t)),axis=1) for sig in range(3)]).astype(np.float64)
    payload=dict(data=data,tx_list=s.CLASSES,rx_list=s.MAIN_RX_NAMES,capture_date_list=['2021_03_01','2021_03_08','2021_03_15','2021_03_23'],equalized_list=[0,1])
    def raw(*a):events.append('raw-whole-pkl');return payload
    mp.setattr(m,'_load_raw',raw)
    class Reader(torch.nn.Module):
        def __init__(self):super().__init__();self.register_buffer('fixed',torch.tensor(1.))
        def features(self,x):events.append(('F',len(x)));return x.mean((1,2))[:,None].repeat(1,160)
        def classify_features(self,x):return x[:,:6]
    class Restore(torch.nn.Module):
        def __init__(self):super().__init__();self.register_buffer('fixed',torch.tensor(.75))
        def forward(self,x):events.append(('G',len(x)));return x*self.fixed
    reader=Reader().eval().requires_grad_(False);g=Restore().eval().requires_grad_(False)
    b=dict(reader=reader,dura=(g,lambda x:(g(x),)),reader_identity=dict(model='original_F0_E50',parameters=232993,epoch=50,updates=11100,model_seed=2026092701),
        dura_identity=dict(deployment_id='dura_v1',parameters=266951,epoch=20,updates=4940,precision='FP32',TF32=False))
    mp.setattr(m,'_backend',lambda *a,**kw:b)
    original=m._observe;observed=[];lookup={sid(*r):r for r in records}
    def observe(raw,**kwargs):
        r=lookup[kwargs['physical_id']];np.testing.assert_array_equal(raw,payload['data'][r[0]][r[1]][r[2]][r[3]][r[4]].T)
        assert raw.dtype==np.float64 and kwargs['receiver']==s.MAIN_RX_NAMES[r[1]] and kwargs['day']==payload['capture_date_list'][r[2]]
        observed.append(kwargs['physical_id']);return original(raw,**kwargs)
    mp.setattr(m,'_observe',observe)
    return m,c,meta,contract,payload,b,events,observed


def test_actual_toy_build_single_raw_observation_G_and_public_readonly_contract(tmp_path,monkeypatch):
    m,c,meta,contract,payload,b,events,observed=fixture(tmp_path,monkeypatch);done=m.build(c,'cpu');root=Path(c['output_root'])/'U_s'
    assert events[:4]==[('G',2),('F',2),('F',2),'metadata'] and events[4]=='raw-whole-pkl'
    np.testing.assert_array_equal(observed,meta['ids']);assert len(observed)==30 and done['original_U_s_reads']==done['G_forward_rows']==30 and done['real_F_forward_rows']==0
    assert done['Us_labels_read']==done['private_truth_reads']==done['target_reads']==0 and done['original_models_unchanged']
    assert {p.name for p in root.iterdir()}==m.PUBLIC_FILES
    a=m.load_public(root,m.CAPSULE_ID,m.SPLIT_ID,30)
    try:
        np.testing.assert_array_equal(a['ids'],meta['ids']);np.testing.assert_allclose(a['restored'],.75*a['received'],atol=0,rtol=0)
        assert not a['restored'].flags.writeable and not a['received'].flags.writeable
        assert set(a['metadata'])=={'rx_i','day_i','domain'} and set(a['metadata']['domain'])==set(range(15))
        assert not ({'y','tx','eq_i','sig_i','base_index','raw_mainsig'}&set(a['manifest']))
        assert a['manifest']['parent_F0_identity']==b['reader_identity'] and a['manifest']['data_status']=='VALIDATED_ONCE'
    finally:m.close_public(a)
    assert a['restored']._mmap.closed
    # Public loader is independent of private builder audit/config/completion.
    (root.parent/'complete.json').unlink();a=m.load_public(root,m.CAPSULE_ID,m.SPLIT_ID);m.close_public(a)
    with pytest.raises(FileExistsError):m.build(c,'cpu')


def test_labels_wrong_role_or_bad_raw_never_complete(tmp_path,monkeypatch,subtests):
    m,c,meta,contract,payload,b,events,observed=fixture(tmp_path,monkeypatch);folder=Path(c['source_config']['source_cache']);original=(folder/'U_s_metadata.npz').read_bytes()
    for defect in ('y','wrong-role','raw-layout'):
        with subtests.test(defect=defect):
            (folder/'U_s_metadata.npz').write_bytes(original);events.clear();c['output_root']=str(tmp_path/defect/'cache')
            if defect=='y':np.savez(folder/'U_s_metadata.npz',**meta,y=np.zeros(30,np.int64))
            elif defect=='wrong-role':bad=deepcopy(meta);bad['ids'][0]=contract['role_ids']['L_s'][0];np.savez(folder/'U_s_metadata.npz',**bad)
            else:
                r=next(r for r in [(tx,rx,d,1,1) for tx in (0,1) for rx in m.source.RECEIVERS for d in (1,2,3)] if sid(*r)==meta['ids'][0]);payload['data'][r[0]][r[1]][r[2]][1]=np.zeros((3,255,2),np.float64)
            with pytest.raises(ValueError):m.build(c,'cpu')
            assert not Path(c['output_root'],'complete.json').exists()
            if defect!='raw-layout':assert 'raw-whole-pkl' not in events


def test_public_privacy_wrong_handles_and_output_collision(tmp_path,monkeypatch,subtests):
    m,c,meta,contract,payload,b,events,observed=fixture(tmp_path,monkeypatch)
    for delta in (dict(truth='forbidden'),dict(batch_size=True),dict(output_root=c['source_config']['source_cache'])):
        with subtests.test(delta=delta),pytest.raises(ValueError):m.validate_config(dict(c,**delta))
    m.build(c,'cpu');folder=Path(c['output_root'])/'U_s'
    with pytest.raises(ValueError):m.load_public(folder,'wrong',m.SPLIT_ID)
    np.save(folder/'y.npy',np.zeros(30,np.int64))
    with pytest.raises(ValueError):m.load_public(folder,m.CAPSULE_ID,m.SPLIT_ID)


def test_smoke_or_frozen_model_failure_never_opens_or_completes_source(tmp_path,monkeypatch):
    m,c,meta,contract,payload,b,events,observed=fixture(tmp_path,monkeypatch)
    monkeypatch.setattr(m,'_smoke',lambda *a:dict(status='FAIL'))
    with pytest.raises(ValueError):m.build(c,'cpu')
    assert events==[] and not Path(c['output_root']).exists()
