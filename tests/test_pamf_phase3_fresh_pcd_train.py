"""Fresh U, original source replay, one actual synthetic PCD student."""
from contextlib import contextmanager
import importlib
import json
from pathlib import Path
import numpy as np
import pytest
import torch


def api():return importlib.import_module('experiments.pamf_phase3_open_world.fresh_pcd_train')


@contextmanager
def fixture(root,mp):
    from test_pamf_phase3_star_common_transport_train import fixture as controls
    with controls(root,mp) as (_,old,pm,b,U,events,_):
        m=api();mp.setattr(m,'STEPS',2);mp.setattr(m,'U_COUNT',16);mp.setattr(m,'U_BATCH',4);mp.setattr(m,'COUNTS',dict(m.body.COUNTS))
        c=dict(run_id=m.RUN_ID,code_commit='d'*40,reference_config=old['reference_config'],U_public_root=str(root/'fresh/discovery/public'),U_capsule_id='p3fresh-U-cap',U_split_id='fresh-U-split',U_expected_count=16,output_root=str(root/'fresh_train'),steps=2,lr=.0001)
        public=Path(c['U_public_root']);public.mkdir(parents=True);ids=np.asarray([f'p3fresh-{i:024x}' for i in range(16)])
        manifest=dict(schema='phase3_min_v1',data_status='VALIDATED_ONCE',capsule_id=c['U_capsule_id'],split_id=c['U_split_id'],count=16,iq_shape=[16,2,256],registered_classes=m.body.base.native.CLASSES,processing_route='residual',mode='post_sync',equalization_enabled=False,fs_hz=25000000,single_observation=True)
        (public/'manifest.json').write_text(json.dumps(manifest),encoding='utf-8');np.save(public/'ids.npy',ids);np.save(public/'received.npy',U[2])
        events.clear();opened=m._public
        def public_U(config):
            assert Path(c['output_root'],'smoke.json').exists();events.append('fresh_U');return opened(config)
        mp.setattr(m,'_public',public_U)
        yield m,c,pm,b,events,manifest


def test_fresh_PCD_actual_IQ_source_indices_and_strict_single_checkpoint(tmp_path,monkeypatch,subtests):
    with fixture(tmp_path,monkeypatch) as (m,c,pm,b,events,manifest):
        c['output_root']=str(tmp_path/'fresh-run/train')
        done=m.train(c,'cpu');assert events==['checkpoint','L','OE','fresh_U']
        assert done['source_sampling_matched'] and done['source_control_source_only'] and done['target_U_reads']==16 and done['CCD_updates']==0
        assert done['real_parent_signal_forward_rows']==sum(m.COUNTS.values())+16 and done['real_student_signal_forward_rows']==(256+16+8)*2
        old=Path(c['reference_config']['train_config']['output_root'])/'SOURCE/steps.jsonl'
        for new,source in zip(Path(c['output_root'],'steps.jsonl').read_text().splitlines(),old.read_text().splitlines()):
            new,source=json.loads(new),json.loads(source);assert new['L_indices']==source['L_indices'] and new['OE_indices']==source['OE_indices']
        launch=importlib.import_module('experiments.pamf_phase3_open_world.fresh_world_launch');read=launch._read
        def tiny_budget(path):
            actual=read(path);assert actual['updates']==actual['total_optimizer_updates']==2
            # Only the two fixed-production budget counters are adapted; every
            # terminal flag is read from the actual synthetic training output.
            return dict(actual,updates=444,total_optimizer_updates=444)
        monkeypatch.setattr(launch,'_read',tiny_budget)
        handed=launch._terminal(dict(run_root=str(tmp_path/'fresh-run'),code_commit=c['code_commit']),'train')
        assert handed['source_control_frozen_before_training'] is True and read(Path(c['output_root'],'complete.json'))==done
        original_loader=m.body.base.load_trained_model
        def rebuilt_parent(*a,**kw):
            model,identity=original_loader(*a,**kw);model.register_buffer('runtime_cache',torch.ones(1),persistent=False);return model,identity
        monkeypatch.setattr(m.body.base,'load_trained_model',rebuilt_parent)
        model,i=m.load_trained_model(c,'cpu');assert i==done['identity'] and i['target_U_reads']==16 and not i['source_only']
        assert all(not p.requires_grad for p in model.parameters()) and done['all_student_Sinc_unchanged'] and done['all_student_buffers_unchanged']
        assert 'runtime_cache' in dict(model.named_buffers()) and 'runtime_cache' not in model.state_dict()
        cp=Path(c['output_root'],'final.pt');saved=cp.read_bytes()
        sinc=next(k for k in dict(model.named_parameters()) if k.startswith(m.body.signalcore.SINC_PREFIX))
        buffer=next(k for k in dict(model.named_buffers()) if k in model.state_dict() and not k.startswith(m.body.signalcore.SINC_PREFIX))
        for defect in ('Sinc','buffer','ancestry'):
            with subtests.test(defect=defect):
                payload=torch.load(cp,weights_only=False)
                if defect=='ancestry':payload['parent_signal_identity']=dict(wrong_parent=True)
                else:payload['network'][sinc if defect=='Sinc' else buffer]+=1
                torch.save(payload,cp)
                with pytest.raises(ValueError):m.load_trained_model(c,'cpu')
                cp.write_bytes(saved)


def test_U_role_and_source_trace_negative_never_complete(tmp_path,monkeypatch,subtests):
    with fixture(tmp_path,monkeypatch) as (m,c,pm,b,events,manifest):
        for delta in (dict(builder_config={}),dict(unknown_count=3),dict(U_expected_count=18),dict(U_public_root=str(tmp_path/'future_clean/public'))):
            with subtests.test(delta=delta),pytest.raises(ValueError):m.validate_config(dict(c,**delta))
        path=Path(c['reference_config']['train_config']['output_root'])/'SOURCE/steps.jsonl';rows=path.read_text().splitlines();r=json.loads(rows[0]);r['L_indices'][0]=-1;rows[0]=json.dumps(r);path.write_text('\n'.join(rows)+'\n')
        with pytest.raises(ValueError,match='sampling'):m.train(c,'cpu')
        assert not Path(c['output_root'],'complete.json').exists()
