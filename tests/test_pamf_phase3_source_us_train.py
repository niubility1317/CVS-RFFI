"""Source Us replay keeps matched source streams, frozen Sinc and hidden TX."""
from contextlib import contextmanager
import importlib
import json
from pathlib import Path
import numpy as np
import pytest
import torch
from torch.nn import functional as F


def api():return importlib.import_module('experiments.pamf_phase3_open_world.source_us_train')


def test_permutation_carries_tail_without_losing_source_records():
    m=api();sampler=m.PermutationStream(56700,2026103150)
    actual=np.concatenate([sampler.sample(128) for _ in range(444)])
    rng=np.random.default_rng(2026103150);expected=np.concatenate((rng.permutation(56700),rng.permutation(56700)))[:56832]
    np.testing.assert_array_equal(actual,expected)
    assert len(actual)==56832 and len(set(actual.tolist()))==56700
    assert set(actual[:56700])==set(range(56700))


def test_masked_PL_uses_whole_batch_and_fixed_six_training_prototypes():
    m=api();g=torch.Generator().manual_seed(8);z=F.normalize(torch.randn(4,160,generator=g),dim=1).requires_grad_()
    prototypes=F.normalize(torch.randn(6,160,generator=g),dim=1).requires_grad_();pseudo=torch.tensor([0,1,2,3])
    confidence=torch.tensor([.95,.949,.99,.2],dtype=torch.float64)
    value=m.pseudo_loss(z,pseudo,confidence,prototypes)
    expected=(F.cross_entropy(z@prototypes.detach().T/.1,pseudo,reduction='none')*torch.tensor([1.,0.,1.,0.])).mean()
    torch.testing.assert_close(value['loss'],expected);value['loss'].backward()
    assert value['accepted']==2 and value['count']==4 and z.grad[0].norm()>0 and z.grad[2].norm()>0
    assert torch.equal(z.grad[[1,3]],torch.zeros_like(z.grad[[1,3]])) and prototypes.grad is None
    labels=torch.repeat_interleave(torch.arange(6),2);centers=m.training_prototypes(torch.eye(160)[:12],labels)
    assert centers.shape==(6,160) and torch.allclose(centers.norm(dim=1),torch.ones(6))
    with pytest.raises(ValueError):m.pseudo_loss(z,pseudo+6,confidence,prototypes)


@contextmanager
def fixture(root,mp):
    from test_pamf_phase3_star_common_transport_train import fixture as controls
    with controls(root,mp) as (_,old,pm,b,U,events,_):
        m=api();mp.setattr(m,'STEPS',2);mp.setattr(m,'US_COUNT',200);mp.setattr(m,'COUNTS',dict(m.body.COUNTS))
        c=dict(run_id=m.RUN_ID,code_commit='b'*40,reference_config=old['reference_config'],Us_public_root=str(root/'Us_cache/U_s'),
            Us_capsule_id=m.US_CAPSULE_ID,Us_split_id=m.US_SPLIT_ID,Us_expected_count=200,output_root=str(root/'Us_train'),steps=2,lr=.0001)
        rng=np.random.default_rng(17);iq=rng.normal(size=(200,2,256)).astype(np.float32);ids=np.asarray(['Us-'+str(i) for i in range(200)])
        manifest=dict(schema=m.US_SCHEMA,status=m.US_STATUS,run_id=m.RUN_ID,role='U_s',count=200,data_status='VALIDATED_ONCE',capsule_id=m.US_CAPSULE_ID,split_id=m.US_SPLIT_ID,
            input_path='cached_P1_only',received_input_path='P0',restored_input_path='P1',source_only=True,parent_F0_identity=b['reader_identity'],dura_identity=b['dura_identity'])
        meta=dict(rx_i=np.resize(np.asarray([1,3,4,6,8],np.int64),200),day_i=np.resize(np.asarray([1,2,3],np.int64),200))
        meta['domain']=np.searchsorted([1,3,4,6,8],meta['rx_i'])*3+meta['day_i']-1
        bundle=dict(manifest=manifest,ids=ids,received=iq,restored=iq.copy(),metadata=meta)
        def public(config):
            assert Path(c['output_root'],'smoke.json').exists();events.append('Us');return bundle
        mp.setattr(m,'_public',public);events.clear()
        yield m,c,pm,b,events,bundle


def test_actual_source_replay_teacher_PL_strict_reload_and_no_Us_read_on_load(tmp_path,monkeypatch,subtests):
    with fixture(tmp_path,monkeypatch) as (m,c,pm,b,events,bundle):
        from experiments.pamf_phase3_open_world import source_us_cache as cache
        monkeypatch.setattr(cache,'COUNT',200);c['output_root']=str(tmp_path/'handoff/train')
        public=Path(c['Us_public_root']);public.mkdir(parents=True)
        bundle['ids']=np.asarray([f'{i+1:024x}' for i in range(200)])
        manifest=dict(bundle['manifest'],code_commit=c['code_commit'],iq_shape=[200,2,256],precision='FP32',files=sorted(cache.PUBLIC_FILES),metadata_fields=sorted(cache.METADATA_FIELDS),
            episode=cache.source.EPISODE,receiver_seed=2027,second_RMS_applied=False,Us_labels_read=0,
            source_contract_handle=dict(classes=m.body.base.native.CLASSES,source_rxs=[1,3,4,6,8],source_days=[1,2,3],split_seed=392005,counts=dict(L_s=360,U_s=200,V=180)))
        (public/'manifest.json').write_text(json.dumps(manifest),encoding='utf-8');np.save(public/'ids.npy',bundle['ids']);np.save(public/'received.npy',bundle['received']);np.save(public/'restored.npy',bundle['restored']);np.savez(public/'metadata.npz',**bundle['metadata'])
        def actual_public(config):
            assert Path(c['output_root'],'smoke.json').is_file();events.append('Us')
            return cache.load_public(config['Us_public_root'],config['Us_capsule_id'],config['Us_split_id'],config['Us_expected_count'])
        monkeypatch.setattr(m,'_public',actual_public)
        monkeypatch.setattr(torch,'from_numpy',lambda *a:pytest.fail('NumPy ABI bridge forbidden'))
        monkeypatch.setattr(torch.Tensor,'numpy',lambda *a:pytest.fail('NumPy ABI bridge forbidden'))
        before=m.body._snapshot(b['model']);done=m.train(c,'cpu');out=Path(c['output_root'])
        assert events==['checkpoint','L','OE','Us'] and done['source_sampling_matched'] and done['source_only']
        assert done['Us_ground_truth_reads']==done['target_U_reads']==done['target_Q_reads']==done['private_truth_reads']==0
        assert done['source_Us_sampled_exposures']==256 and done['source_Us_unique_records']==200
        assert done['real_parent_signal_forward_rows']==sum(m.COUNTS.values()) and done['real_F_forward_rows']==200
        assert done['real_student_signal_forward_rows']==(192+64+128)*2
        rows=[json.loads(x) for x in (out/'steps.jsonl').read_text().splitlines()]
        controls=[json.loads(x) for x in (Path(c['reference_config']['train_config']['output_root'])/'SOURCE/steps.jsonl').read_text().splitlines()]
        assert all(a['L_indices']==z['L_indices'] and a['OE_indices']==z['OE_indices'] for a,z in zip(rows,controls))
        assert len(set(rows[0]['Us_indices']+rows[1]['Us_indices']))==200 and all(len(a['Us_indices'])==128 for a in rows)
        assert rows[0]['gradient_norms']['body']>0 and rows[0]['gradient_norms']['last']>0
        assert done['all_student_Sinc_buffers_unchanged'] and done['student_body_changed'] and done['student_projection_changed']
        assert all(torch.equal(v,b['model'].state_dict()[k].detach().cpu()) for k,v in before.items())
        from experiments.pamf_phase3_open_world import source_us_launch as launch
        read=launch._read
        def tiny_budget(path):
            actual=read(path);assert actual['updates']==actual['total_optimizer_updates']==2
            return dict(actual,updates=444,total_optimizer_updates=444)
        monkeypatch.setattr(launch,'_read',tiny_budget)
        assert launch._terminal(dict(run_root=str(tmp_path/'handoff'),code_commit=c['code_commit']),'train')['source_only']
        original_loader=m.body.base.load_trained_model
        def parent(*a,**kw):
            model,i=original_loader(*a,**kw);model.register_buffer('runtime_only',torch.ones(1),persistent=False);return model,i
        monkeypatch.setattr(m.body.base,'load_trained_model',parent)
        monkeypatch.setattr(m,'_public',lambda *a:pytest.fail('Loader must never read Us values'))
        model,identity=m.load_trained_model(c,'cpu');assert identity==done['identity'] and all(not p.requires_grad for p in model.parameters())
        assert identity['source_only'] and identity['source_Us_count']==200 and identity['identity_head_updates']==0
        assert all(not mod.training for mod in model.modules()) and 'runtime_only' not in model.state_dict()
        cp=out/'final.pt';saved=cp.read_bytes();network=torch.load(cp,weights_only=False)['network']
        sinc=next(k for k in network if k.startswith(m.body.signalcore.SINC_PREFIX));buffer=next(k for k in dict(model.named_buffers()) if k in network and not k.startswith(m.body.signalcore.SINC_PREFIX))
        for defect in ('Sinc','buffer','ancestry'):
            with subtests.test(defect=defect):
                payload=torch.load(cp,weights_only=False)
                if defect=='ancestry':payload['parent_signal_identity']={'wrong':True}
                else:payload['network'][sinc if defect=='Sinc' else buffer]+=1
                torch.save(payload,cp)
                with pytest.raises(ValueError):m.load_trained_model(c,'cpu')
                cp.write_bytes(saved)
        with pytest.raises(FileExistsError):m.train(c,'cpu')


def test_config_control_and_Us_private_fields_fail_without_complete(tmp_path,monkeypatch,subtests):
    with fixture(tmp_path,monkeypatch) as (m,c,pm,b,events,bundle):
        for delta in (dict(truth='private'),dict(U_public_root='target'),dict(Us_expected_count=201),dict(steps=1),dict(output_root=c['Us_public_root']),dict(Us_public_root=str(tmp_path/'V'))):
            with subtests.test(delta=delta),pytest.raises(ValueError):m.validate_config(dict(c,**delta))
        path=Path(c['reference_config']['train_config']['output_root'])/'SOURCE/steps.jsonl';original=path.read_text();lines=original.splitlines();row=json.loads(lines[0]);row['L_indices'][0]=-1;lines[0]=json.dumps(row);path.write_text('\n'.join(lines)+'\n')
        with pytest.raises(ValueError,match='sampling'):m.train(c,'cpu')
        assert not Path(c['output_root'],'complete.json').exists()
    # Wrong public label fields are rejected before teacher inference or updates.
    with fixture(tmp_path/'second',monkeypatch) as (m,c,pm,b,events,bundle):
        bundle['metadata']['y']=np.zeros(200,np.int64)
        monkeypatch.setattr(m,'_teacher',lambda *a:pytest.fail('Illegal Us must not reach teacher'))
        with pytest.raises(ValueError,match='Us'):m.train(c,'cpu')
        assert not Path(c['output_root'],'complete.json').exists()
