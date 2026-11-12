import json
import numpy as np
import torch
import pytest

from experiments.pamf_phase2_practical import clean_tweak_lora_worker as worker
from experiments.pamf_phase2_practical import external_additional as extra


@pytest.mark.parametrize('method',['lora','orthogonal'])
def test_tweak_registration_keeps_single_query_decisions(tmp_path,monkeypatch,method):
    cap=tmp_path/'data/capsules/P01/MIX01/1-1';(cap/'splits').mkdir(parents=True)
    (cap/'manifest.json').write_text(json.dumps({'capsule_id':'cap'}))
    for n in (0,2):
        spec=dict(k=1,new_count=n,split_id=f's{n}',registered_classes=list(range(6+n)))
        (cap/'splits'/f'{n}.json').write_text(json.dumps(spec))
    rng=np.random.default_rng(3);bank=rng.standard_normal((8,2,256)).astype(np.float32)
    def task(cap,split_id,*,stage,clean_root,task_root):
        n=int(split_id[1:]);count=6+n
        return dict(support_x=bank[:count],support_y=np.arange(count),support_ids=np.arange(count),query_x=bank[:count])
    model=extra.NativeModel('tweak_m1',6,5).eval()
    bundle=dict(model=model,profile='tweak_m1',classes=tuple(range(6)),resources={},native_state={})
    monkeypatch.setattr(worker,'activate',lambda rt:None);monkeypatch.setattr(worker,'check',lambda *args:None)
    monkeypatch.setattr(worker.extra,'load_source',lambda *args:bundle)
    monkeypatch.setattr(worker,'load_task',task)
    monkeypatch.setattr(worker,'split_rows',lambda cap,spec:np.arange(len(spec['registered_classes'])))
    if method=='orthogonal':
        from experiments.pamf_phase2_practical.group2_cil_bridge import register
        monkeypatch.setattr(worker,'register_c',lambda prepared,x,y,seed,methods:
                            {'orthogonal':register(prepared['orthogonal'],x,y,seed=seed,epochs=1)})
    actual=worker.score_fixed;observed=[]
    def score(*args,**kwargs):observed.append(kwargs['batch_size']);return actual(*args,**kwargs)
    monkeypatch.setattr(worker,'score_fixed',score)
    rt=dict(clean_root='unused',data_root=str(tmp_path/'data'))
    job=dict(profile='tweak_m1',methods=[method],seed=2026092701,capsule='capsules/P01/MIX01/1-1',
             output=str(tmp_path/'out'),points=[[1,2]])
    worker.run(rt,job)
    result=json.loads((tmp_path/'out/predictions.json').read_text())['rows'][0]
    assert observed==[1] and result['query_records_per_decision']==1
    if method=='lora':
        assert result['knn_k']==1 and result['claim']=='Tweak_M1_B_to_LoRa_core_transplant'
    else:
        assert result['knn_k'] is None and result['claim']=='Tweak_M1_B_to_Orthogonal_core_transplant'
