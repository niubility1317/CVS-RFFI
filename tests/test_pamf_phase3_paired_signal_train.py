"""Synthetic same-record pair model and matched three-student training."""
from contextlib import contextmanager
import importlib
import json
from pathlib import Path
import numpy as np
import pytest
import torch
from torch.nn import functional as F


def api():
    try:return importlib.import_module('experiments.pamf_phase3_open_world.paired_signal_train')
    except ModuleNotFoundError:pytest.fail('paired signal training is not implemented')


class TinyBase(torch.nn.Module):
    def __init__(self):
        super().__init__();self.linear=torch.nn.Linear(512,160);self.calls=0
    def forward(self,x):
        self.calls+=1;return F.normalize(self.linear(x.flatten(1)),dim=1)


def test_zero_initial_pair_delta_bound_and_shared_encoder_gradient():
    m=api();torch.manual_seed(8);base=TinyBase();model=m.PairedSignal(base,True)
    raw=torch.randn(4,2,256);restored=torch.randn(4,2,256)
    initial=model.features(raw,restored)
    assert base.calls==2 and torch.count_nonzero(initial['delta'])==0
    torch.testing.assert_close(initial['z'],base(restored),rtol=1e-6,atol=1e-7)
    with torch.no_grad():model.adapter.bias.fill_(100)
    result=model.features(raw,restored)
    assert float(result['delta'].detach().norm(dim=1).max())<=.10000002
    result['z'][:,0].sum().backward()
    assert base.linear.weight.grad is not None and torch.isfinite(base.linear.weight.grad).all()


def test_p1_control_never_reads_received_even_for_validation_or_gradients():
    m=api();torch.manual_seed(9);model=m.PairedSignal(TinyBase(),False)
    with torch.no_grad():model.adapter.weight.normal_(0,.1)
    raw=torch.randn(4,2,256,requires_grad=True);p1=torch.randn(4,2,256,requires_grad=True)
    first=model(raw,p1);second=model(torch.full_like(raw,float('nan')),p1);third=model(None,p1)
    assert torch.equal(first,second) and torch.equal(first,third)
    first[:,0].sum().backward();assert raw.grad is None and p1.grad is not None
    assert not any(x.training for x in model.modules())


@contextmanager
def fixture(root,mp):
    from test_pamf_phase3_momentum_prototype_train import fixture as previous
    with previous(root,mp) as (_,c,pm,b,events):
        m=api();mp.setattr(m,'STEPS',2);mp.setattr(m,'U_COUNT',16);mp.setattr(m,'U_BATCH',4);mp.setattr(m,'COUNTS',dict(m.body.COUNTS))
        old_cache=m.body._source_cache
        def paired_source_fixture(*args):
            banks,manifest,roles=old_cache(*args)
            for role in banks:roles[role]['received']=.91*banks[role]['iq']
            return banks,manifest,roles
        mp.setattr(m.body,'_source_cache',paired_source_fixture)
        c=dict(c,run_id=m.RUN_ID,output_root=str(root/'pair_train'))
        yield m,c,pm,b,events


def test_three_arms_same_parent_source_stream_and_sources_frozen_before_u(tmp_path,monkeypatch):
    with fixture(tmp_path,monkeypatch) as (m,c,pm,b,events):
        original=m._load_U
        def checked(*args):
            for arm in m.ARMS[:2]:assert Path(c['output_root'],arm,'complete.json').is_file()
            return original(*args)
        monkeypatch.setattr(m,'_load_U',checked)
        done=m.train(c,'cpu');out=Path(c['output_root'])
        assert done['total_optimizer_updates']==6 and done['SOURCE_completed_before_U_open']
        assert done['source_control_frozen_before_training'] and done['source_sampling_matched']
        assert done['all_adapters_changed'] and done['G_forward_rows']==done['real_F_forward_rows']==0
        assert done['real_parent_signal_forward_rows']==sum(m.COUNTS.values())
        rows={a:[json.loads(x) for x in (out/a/'steps.jsonl').read_text().splitlines()] for a in m.ARMS}
        for i in range(2):
            assert all(rows[a][i]['L_indices']==rows[m.ARMS[0]][i]['L_indices'] for a in m.ARMS)
            assert all(rows[a][i]['OE_indices']==rows[m.ARMS[0]][i]['OE_indices'] for a in m.ARMS)
            assert not rows['SOURCE_P1'][i]['U_indices'] and not rows['SOURCE_PAIR'][i]['U_indices']
            assert len(rows['TARGET_PAIR'][i]['U_indices'])==4
            assert all(rows[a][i]['gradient_norms']['adapter']>0 for a in m.ARMS)
        assert done['target_U_P0_reads']==16 and done['paired_augmentations_synchronized']
        for arm in m.ARMS:
            model,identity=m.load_trained_model(c,arm,'cpu')
            assert identity==done['identities'][arm] and identity['signal_parameters']==386081
            assert model.paired==(arm!='SOURCE_P1') and all(not p.requires_grad for p in model.parameters())
        cp=out/'TARGET_PAIR/final.pt';value=torch.load(cp,weights_only=False)
        key=next(k for k in value['network'] if k.startswith('base.'+m.body.signalcore.SINC_PREFIX))
        value['network'][key]+=1;torch.save(value,cp)
        with pytest.raises(ValueError,match='Sinc'):m.load_trained_model(c,'TARGET_PAIR','cpu')


def test_p0_p1_phase_gain_replay_is_synchronized_and_rejects_different_p1(tmp_path,monkeypatch):
    with fixture(tmp_path,monkeypatch) as (m,c,pm,b,events):
        t=c['reference_config']['train_config'];manifest=json.loads((Path(t['output_root'])/'U_manifest.json').read_text())
        U=m._load_U(c,manifest,set())
        assert U['P0_views'].shape==U['P1_views'].shape==(3,16,2,256)
        for source in ('P0_views','P1_views'):
            generator=torch.Generator().manual_seed(m.body.AUG_SEED)
            raw=m.body.base._tensor(U[source][0],'cpu')
            for j in (1,2):np.testing.assert_array_equal(U[source][j],m.body.views.phase_gain_view(raw,generator).tolist())
        cache=Path(t['output_root'])/'U_features.npz'
        with np.load(cache,allow_pickle=False) as z:values={k:z[k].copy() for k in z.files}
        values['views'][1,0,0,0]+=1;np.savez(cache,**values)
        with pytest.raises(ValueError,match='synchron'):m._load_U(c,manifest,set())


def test_private_inputs_and_partial_source_never_publish_full_or_load(tmp_path,monkeypatch):
    with fixture(tmp_path,monkeypatch) as (m,c,pm,b,events):
        for extra in (dict(truth='forbidden'),dict(Q_root='forbidden'),dict(steps=1)):
            with pytest.raises(ValueError):m.validate_config(dict(c,**extra))
        monkeypatch.setattr(m,'_load_U',lambda *a:(_ for _ in ()).throw(ValueError('U interrupted')))
        with pytest.raises(ValueError,match='interrupted'):m.train(c,'cpu')
        assert Path(c['output_root'],'SOURCE_P1/final.pt').is_file() and Path(c['output_root'],'SOURCE_PAIR/final.pt').is_file()
        assert not Path(c['output_root'],'complete.json').exists()
        with pytest.raises(ValueError):m.load_trained_model(c,'SOURCE_P1','cpu')
