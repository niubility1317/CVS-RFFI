import numpy as np
import pytest
import torch

from experiments.phase1_dura_orthogonal.source_diverse_views import diverse_epochs
from experiments.phase1_dura_orthogonal import train_source


def test_schedule_covers_all_30_views_twice_with_equal_training_budget():
    planned=[diverse_epochs(e) for e in range(1,21)]
    assert planned[:10]==planned[10:]
    assert sorted(e for group in planned for e in group)==sorted(list(range(1,31))*2)
    assert 20*3*32768==1966080
    assert 20*(3*32768//32)==61440
    for e in (0,21):
        with pytest.raises(ValueError):diverse_epochs(e)


def test_prepare_uses_requested_views_in_order_with_matched_clean(monkeypatch,tmp_path):
    monkeypatch.setattr(train_source.core,'COUNT',3)
    monkeypatch.setattr(train_source.core,'read',lambda p:dict(count=3,namespace=f'learned-action/train/epoch{p.parent.name}',TX_labels_read=0,F_outputs_read=0,target_reads=0))
    class Group:
        def __enter__(self):return {'scene':np.array(['clean','practical_high','practical_mid'])}
        def __exit__(self,*args):return False
    def fake_load(path,**kwargs):
        if path.name=='groups.npz':return Group()
        epoch=int(path.parent.name)
        return np.stack([np.full((2,256),epoch*10+i,dtype=np.float32) for i in range(3)])
    monkeypatch.setattr(train_source.np,'load',fake_load)
    clean=np.stack([np.full((2,256),i,dtype=np.float32) for i in range(3)])
    data,counts=train_source.prepare(tmp_path,None,clean,tmp_path,np.array([2,0]),'cpu',32,
        architecture='large_direct_timefreq_stable',source_epochs=(4,5,6))
    assert data[0][:,0,0].tolist()==[42.,40.,52.,50.,62.,60.]
    assert data[4][:,0,0].tolist()==[2.,0.,2.,0.,2.,0.]
    assert torch.equal(data[1],data[0]) and torch.count_nonzero(data[2])==0
    assert counts['clean']==3 and counts['practical_mid']==3 and sum(counts.values())==6


def test_ground_rejects_mismatched_training_schedule(tmp_path,monkeypatch):
    import json
    from experiments.phase1_dura_orthogonal import ground
    from experiments.phase1_dura_orthogonal.large_direct_timefreq import WIDTH, LAYERS, HEADS, TOKENS, SCALE
    from experiments.phase1_dura_orthogonal.large_direct_timefreq_stable import StableLargeDirectTimeFreq
    folder=tmp_path/'runs'/'source';folder.mkdir(parents=True)
    common=dict(commit='test',epochs=20,architecture='large_direct_timefreq_stable',
        objective='wide_large_direct_timefreq_stable_diverse',source_fit_size=32768,
        DURA_training_reads=0,DURA_frozen=True,TX_labels_read=0,F_weights_read=0,
        F_outputs_read=0,V_reads=0,target_reads=0,inference_received_iq_only=True,
        direct_timefreq_width=WIDTH,direct_timefreq_layers=LAYERS,direct_timefreq_heads=HEADS,
        direct_timefreq_tokens_per_domain=TOKENS,direct_timefreq_scale=SCALE,
        direct_timefreq_normalization='token_layernorm_groupnorm',
        source_view_schedule=[list(diverse_epochs(e)) for e in range(1,21)],
        source_view_epochs=30,source_view_repeats=2,source_view_epoch_counts=[{}]*20,
        source_view_reload_seconds=[1.]*19,source_counts_scope='first_training_epoch')
    summary=dict(common,status='SOURCE_TRAINED',fit_ids=32768)
    state=dict(common,schema=ground.SCHEMA,fit_ids=np.arange(32768),width=WIDTH,model=StableLargeDirectTimeFreq().state_dict())
    (folder/'source_summary.json').write_text(json.dumps(summary),encoding='utf8');torch.save(state,folder/'final.pt')
    monkeypatch.setattr(ground,'frozen_dura',lambda *a:torch.nn.Identity())
    monkeypatch.setattr(ground.core,'teacher',lambda *a:(torch.nn.Identity(),None))
    assert isinstance(ground.load_fixed(tmp_path,'source','test')[0],StableLargeDirectTimeFreq)
    summary['source_view_schedule'][0]=[21,22,23]
    (folder/'source_summary.json').write_text(json.dumps(summary),encoding='utf8')
    with pytest.raises(ValueError,match='diverse source-view'):
        ground.load_fixed(tmp_path,'source','test')
