"""Two actual synthetic reader trajectories and strict source-only artifacts."""
from contextlib import contextmanager
from copy import deepcopy
import importlib
import json
from pathlib import Path
import numpy as np
import pytest
import torch


def api():return importlib.import_module('experiments.pamf_phase3_open_world.logit_energy_train')


@contextmanager
def fixture(root,mp):
    from test_pamf_phase3_native_identity_train import fixture as old
    with old(root,mp) as (native,oldc,pm,b,parent,pcd,events):
        m=api();mp.setattr(m,'STEPS',2);mp.setattr(m,'COUNTS',dict(L=270,OE=90))
        r03=dict(run_id='20261030-phase3-star-signal-open-s2026092701-r03',arm='E_SIGNAL',signal_parameters=334721)
        pm['model_identities']['E0']=r03;b['model_identity']=r03
        c=dict(oldc,run_id=m.RUN_ID,output_root=str(root/'energy/train'))
        rng=np.random.default_rng(49);roles={};rows=np.asarray([(c,r,d) for c in range(6) for r in (1,3,4,6,8) for d in (1,2,3) for _ in range(3)],np.int64)
        for role,n in m.COUNTS.items():
            iq=rng.normal(size=(n,2,256)).astype(np.float32);domain=np.tile(np.arange(15,dtype=np.int64),n//15)
            meta=dict(rx_i=np.asarray([1,3,4,6,8],np.int64)[domain//3],day_i=domain%3+1,domain=domain)
            if role=='L':meta=dict(meta,y=rows[:,0],rx_i=rows[:,1],day_i=rows[:,2],domain=np.tile(np.repeat(np.arange(15),3),6).astype(np.int64))
            a=dict(ids=np.asarray([role+str(i) for i in range(n)]),restored=iq,metadata=meta,
                   manifest=dict(count=n,parent_F0_identity=pm['parent_F0_identity'],dura_identity=pm['dura_identity']))
            if role=='L':
                with torch.no_grad():value=parent.id_backbone.encoder.classify_features(parent.id_backbone.encoder.features(torch.tensor(iq.tolist())))
                a['teacher_logits']=np.asarray(value.tolist(),np.float32)
            roles[role]=a
        actual_smoke=m._smoke
        def smoke(*args):
            assert not events;value=actual_smoke(*args);events.append('smoke');return value
        mp.setattr(m,'_smoke',smoke)
        def load(path,role):
            assert Path(c['output_root'],'smoke.json').is_file() and events in (['smoke'],['smoke','L'])
            events.append(role);return roles[role]
        mp.setattr(m.cache,'load_role',load)
        yield m,c,pm,b,parent,roles,events


def test_actual_two_arms_only_reader_head_trains_sampling_matched_and_strict_load(tmp_path,monkeypatch,subtests):
    with fixture(tmp_path,monkeypatch) as (m,c,pm,b,parent,roles,events):
        monkeypatch.setattr(torch,'from_numpy',lambda *a:pytest.fail('Use the verified list bridge'))
        monkeypatch.setattr(torch.Tensor,'numpy',lambda *a:pytest.fail('Use the verified list bridge'))
        initial=m._snapshot(parent);done=m.train(c,'cpu');root=Path(c['output_root'])
        assert events==['smoke','L','OE'] and done['total_optimizer_updates']==4 and done['real_student_forward_rows']==1024
        assert done['real_F_forward_rows']==done['G_forward_rows']==done['real_parent_signal_forward_rows']==0
        assert done['cached_L_teacher_logits_reads']==270 and done['all_student_heads_changed'] and done['all_student_bodies_changed']
        assert all(torch.equal(v,parent.state_dict()[k]) for k,v in initial.items())
        traces={a:[json.loads(v) for v in (root/'rows'/a/'steps.jsonl').read_text().splitlines()] for a in m.ARMS}
        assert all(x['L_indices']==y['L_indices'] and x['OE_indices']==y['OE_indices'] for x,y in zip(*traces.values()))
        monkeypatch.setattr(m,'_roles',lambda *a:pytest.fail('Checkpoint loading must not read L/OE values'))
        for arm in m.ARMS:
            model,i=m.load_trained_model(c,arm,'cpu');assert i['reader_parameters']==232993 and i['trainable_parameters']==232945
            assert i['default_head_parameters']==960 and i['identity_head_updates']==2 and i['source_only']
            assert all(not p.requires_grad for p in model.parameters()) and not model.training
            assert all(k.startswith('reader.') for k in model.state_dict())
        with pytest.raises(FileExistsError):m.train(c,'cpu')
        path=root/'rows/RXPAIR/final.pt';payload=torch.load(path,map_location='cpu',weights_only=False)
        for defect in ('parent','head','Sinc','buffer'):
            with subtests.test(defect=defect):
                bad=deepcopy(payload)
                if defect=='parent':bad['original_parent_identity']={'wrong':'ancestor'}
                else:
                    key=m.core.HEAD_WEIGHT if defect=='head' else next(k for k in bad['network'] if k.startswith(m.core.SINC_PREFIX)) if defect=='Sinc' else next(k for k in bad['network'] if k.endswith('running_mean'))
                    if defect=='head':del bad['network'][key]
                    else:bad['network'][key]=bad['network'][key]+1
                torch.save(bad,path)
                with pytest.raises(ValueError):m.load_trained_model(c,'RXPAIR','cpu')
        torch.save(payload,path)


def test_forbidden_inputs_bad_ancestor_and_failed_smoke_before_source_values(tmp_path,monkeypatch,subtests):
    with fixture(tmp_path,monkeypatch) as (m,c,pm,b,parent,roles,events):
        for delta in (dict(truth='private'),dict(U_root='forbidden'),dict(K=3),dict(steps=1)):
            with subtests.test(delta=delta),pytest.raises(ValueError):m.validate_config(dict(c,**delta))
        smoke=m._smoke;monkeypatch.setattr(m,'_smoke',lambda *a:(_ for _ in ()).throw(ValueError('actual smoke failed')))
        with pytest.raises(ValueError,match='actual smoke failed'):m.train(c,'cpu')
        assert not events and not Path(c['output_root']).exists()
        monkeypatch.setattr(m,'_smoke',smoke);b['reader_identity']={'wrong':'F0'}
        with pytest.raises(ValueError):m.train(c,'cpu')
        assert not events and not Path(c['output_root']).exists()


def test_invalid_source_teacher_or_OE_true_label_metadata_rejected(tmp_path,monkeypatch,subtests):
    with fixture(tmp_path,monkeypatch) as (m,c,pm,b,parent,roles,events):
        # Direct source-boundary checks run only with synthetic role bundles.
        monkeypatch.setattr(m.cache,'load_role',lambda path,role:roles[role])
        for defect in ('teacher','OE_y','F0'):
            original=deepcopy(roles)
            with subtests.test(defect=defect),pytest.raises(ValueError):
                if defect=='teacher':roles['L']['teacher_logits']=roles['L']['teacher_logits'][:1]
                elif defect=='OE_y':roles['OE']['metadata']['y']=np.zeros(90,np.int64)
                else:roles['OE']['manifest']['parent_F0_identity']={'wrong':'parent'}
                m._roles(c,pm)
            roles.clear();roles.update(original)
