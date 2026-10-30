"""Synthetic frozen-native cached-source adapter training and checkpoint contracts."""
from copy import deepcopy
import importlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from test_pamf_phase3_star_protected_open_cache import cache_fixture, features, native_config


def api():return importlib.import_module('experiments.pamf_phase3_open_world.star_protected_open_train')


def config(root,arm='R1'):
    t=api();n=native_config(root,arm);n['output_root']=str(root/'train'/arm)
    return dict(run_id=t.RUN_ID,code_commit='a'*40,arm=arm,native_config=n,feature_root=str(root/'features'),
        output_root=n['output_root'],steps=t.STEPS,lr=.0001,sampling_seed=2026103017,slot_seed=2026103018)


def test_sampler_disjoint_coverage_without_replacement_slots_and_identical_L_with_or_without_OE():
    t=api();L=features('L');OE=features('OE');left=t.EpisodeSampler(L,None);right=t.EpisodeSampler(L,OE)
    for step in (1,2,3,4,15,16,90):
        a=left.sample(step);b=right.sample(step)
        for k in ('support','query','pseudo_known_classes'):np.testing.assert_array_equal(a[k],b[k])
        assert len(set(a['support'])|set(a['query']))==192 and not set(a['support'])&set(a['query'])
        assert len(a['pseudo_known_classes'])==2+(step-1)%4 and len(set(a['pseudo_known_classes']))==len(a['pseudo_known_classes'])
        assert a['OE'].size==0 and len(set(b['OE']))==64
        for key in ('support','query'):
            for label in range(6):
                chosen=a[key][L['metadata']['y'][a[key]]==label]
                assert len(chosen)==16 and set(L['metadata']['domain'][chosen])==set(range(15))
        counts=np.bincount(OE['metadata']['domain'][b['OE']],minlength=15)
        assert sorted(counts.tolist())==[4]*11+[5]*4


def test_sampler_rejects_missing_cells_insufficient_unique_records_and_OE_labels(subtests):
    t=api();L=features('L');OE=features('OE')
    for defect in ('L_cell','OE_cell','OE_label','duplicate'):
        l=deepcopy(L);o=deepcopy(OE)
        if defect=='L_cell':l['metadata']['domain'][:4]=14;l['metadata']['rx_i'][:4]=8;l['metadata']['day_i'][:4]=3
        elif defect=='OE_cell':o['metadata']['domain'][:6]=14;o['metadata']['rx_i'][:6]=8;o['metadata']['day_i'][:6]=3
        elif defect=='OE_label':o['metadata']['y']=np.zeros(90,np.int64)
        else:l['ids'][0]=l['ids'][1]
        with subtests.test(defect=defect),pytest.raises(ValueError):t.EpisodeSampler(l,o)


def test_config_rejects_private_paths_wrong_budget_or_parent_row_before_model(tmp_path,monkeypatch,subtests):
    t=api();c=config(tmp_path);t.validate_config(c)
    for delta in (dict(steps=222),dict(arm='P_LOE'),dict(truth='private'),dict(Q_public_root='Q'),dict(feature_root=c['output_root'])):
        with subtests.test(delta=delta),pytest.raises(ValueError):t.validate_config(dict(c,**delta))
    wrong=deepcopy(c);wrong['native_config']['row_id']='R0-s2026092701'
    with pytest.raises(ValueError):t.validate_config(wrong)


@pytest.mark.parametrize('arm',['N0','R0','R1'])
def test_actual_two_step_cached_loop_freezes_native_N0_reads_only_L_and_checkpoint_is_parts_only(tmp_path,monkeypatch,arm):
    with cache_fixture(tmp_path,monkeypatch) as (cache,cc,roles,created,events):
        cache.build(cc,'cpu');t=api();monkeypatch.setattr(t,'STEPS',2);c=config(tmp_path,arm)
        opened=[];real=t.load_features
        def load(root,role):
            assert (Path(c['output_root'])/'smoke.json').is_file()
            if arm=='N0':assert role=='L'
            opened.append(role);return real(root,role)
        monkeypatch.setattr(t,'load_features',load)
        done=t.train(c,'cpu');assert opened==(['L'] if arm=='N0' else ['L','OE'])
        assert done['updates']==done['adapter_updates']==2 and done['native_updates']==0 and done['original_parent_updates']==11100
        assert done['model_native_state_unchanged'] and done['F_forward_rows']==0
        assert done['OE_feature_rows_loaded']==(0 if arm=='N0' else 90)
        assert done['OE_exposures']==(0 if arm=='N0' else 128)
        assert done['trainable_parameters']==(10305 if arm=='R0' else 66177)
        value=torch.load(Path(c['output_root'])/'final.pt',weights_only=False)
        assert 'open_world' in value and 'model' not in value and all(not k.startswith('id_backbone.') for k in value['open_world'])
        restored,identity=t.load_trained_model(c,Path(c['output_root'])/'final.pt','cpu')
        assert identity==done['identity'] and all(not p.requires_grad for p in restored.parameters())
        for name,v in created[-2].open_world.state_dict().items():torch.testing.assert_close(v,restored.open_world.state_dict()[name],atol=0,rtol=0)
        with pytest.raises(FileExistsError):t.train(c,'cpu')
        if arm=='R1':
            bad=deepcopy(value);bad['identity']['native_updates']=1;done['identity']=bad['identity']
            torch.save(bad,Path(c['output_root'])/'final.pt')
            (Path(c['output_root'])/'complete.json').write_text(json.dumps(done),encoding='utf-8')
            with pytest.raises(ValueError,match='identity or source handle differs'):
                t.load_trained_model(c,Path(c['output_root'])/'final.pt','cpu')


def test_gradient_smoke_and_incomplete_identity_fail_before_feature_or_native_reads(tmp_path,monkeypatch):
    with cache_fixture(tmp_path,monkeypatch) as (cache,cc,roles,created,events):
        t=api();c=config(tmp_path)
        monkeypatch.setattr(t,'load_features',lambda *a:(_ for _ in ()).throw(AssertionError('Feature read before smoke')))
        monkeypatch.setattr(t,'_smoke',lambda *a:(_ for _ in ()).throw(ValueError('smoke failed')))
        with pytest.raises(ValueError,match='smoke failed'):t.train(c,'cpu')
        assert not Path(c['output_root']).exists()
        root=Path(c['output_root']);root.mkdir(parents=True);t._write(root/'complete.json',dict(status=t.STATUS,model_native_state_unchanged=False))
        count=len(created)
        with pytest.raises(ValueError):t.load_trained_model(c,root/'final.pt','cpu')
        assert len(created)==count
