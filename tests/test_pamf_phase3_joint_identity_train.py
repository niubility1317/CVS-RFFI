"""Two independent joint students and source-before-U disk contracts."""
from contextlib import contextmanager
import importlib
import json
from pathlib import Path
import pytest
import torch


def api():return importlib.import_module('experiments.pamf_phase3_open_world.joint_identity_train')


@contextmanager
def fixture(root,mp):
    from experiments.pamf_phase3_open_world import fresh_pcd_train as fresh
    public=fresh._public
    from test_pamf_phase3_fresh_pcd_train import fixture as old
    with old(root,mp) as (_,pc,pm,b,events,Umanifest):
        t=api();pc['code_commit']=t.REFERENCE_CODE;mp.setattr(t,'STEPS',2);mp.setattr(t,'U_COUNT',16);mp.setattr(t,'U_BATCH',4);mp.setattr(t,'COUNTS',dict(t.body.COUNTS))
        c=dict(run_id=t.RUN_ID,code_commit='e'*40,reference_config=pc,output_root=str(root/'joint-run/train'),steps=2,lr=.0001)
        def U(config):
            assert Path(c['output_root'],'SOURCE_JOINT/complete.json').exists() and not Path(c['output_root'],'TARGET_JOINT/final.pt').exists()
            events.append('joint_U');return public(config)
        mp.setattr(fresh,'_public',U)
        yield t,c,pm,b,events,Umanifest


def test_actual_two_joint_students_matching_source_then_U_and_strict_reload(tmp_path,monkeypatch,subtests):
    with fixture(tmp_path,monkeypatch) as (t,c,pm,b,events,manifest):
        monkeypatch.setattr(torch,'from_numpy',lambda *a:pytest.fail('No NumPy ABI'));monkeypatch.setattr(torch.Tensor,'numpy',lambda *a:pytest.fail('No NumPy ABI'))
        done=t.train(c,'cpu');out=Path(c['output_root']);assert events==['checkpoint','L','OE','joint_U']
        assert done['source_only_SOURCE'] and done['SOURCE_completed_before_U_open'] and done['total_optimizer_updates']==4
        launch=importlib.import_module('experiments.pamf_phase3_open_world.joint_identity_launch');read=launch._read
        def tiny_budget(path):
            actual=read(path);assert actual['steps_per_arm']==2 and actual['total_optimizer_updates']==4
            return dict(actual,steps_per_arm=444,total_optimizer_updates=888)
        monkeypatch.setattr(launch,'_read',tiny_budget)
        handed=launch._terminal(dict(run_root=str(tmp_path/'joint-run'),code_commit=c['code_commit']),'train')
        assert handed['all_student_heads_changed'] is True and read(out/'complete.json')==done
        assert done['real_parent_signal_forward_rows']==sum(t.COUNTS.values())+16 and done['real_student_signal_forward_rows']==2*256*2+(16+8)*2
        traces={arm:[json.loads(x) for x in (out/arm/'steps.jsonl').read_text().splitlines()] for arm in t.ARMS}
        for s,u in zip(traces['SOURCE_JOINT'],traces['TARGET_JOINT']):
            assert s['L_indices']==u['L_indices'] and s['OE_indices']==u['OE_indices'] and not s['U_indices']
            assert s['L_CE']>0 and s['gradient_norms']['head']>0 and u['gradient_norms']['body']>0 and u['gradient_norms']['last']>0
        for arm in t.ARMS:
            model,i=t.load_trained_model(c,arm,'cpu');assert i==done['identities'][arm] and i['signal_parameters']==335681 and i['identity_head_updates']==2
            assert i['source_only']==(arm=='SOURCE_JOINT') and i['target_U_reads']==(16 if arm=='TARGET_JOINT' else 0) and all(not p.requires_grad for p in model.parameters())
            assert model.joint(torch.randn(2,2,256))['logits'].shape==(2,6)
        cp=out/'TARGET_JOINT/final.pt';saved=cp.read_bytes()
        for defect in ('missing_head','Sinc','ancestry'):
            with subtests.test(defect=defect):
                value=torch.load(cp,weights_only=False)
                if defect=='missing_head':value['network'].pop(t.model_api.HEAD_WEIGHT)
                elif defect=='Sinc':value['network'][next(k for k in value['network'] if k.startswith(t.model_api.SINC_PREFIX))]+=1
                else:value['parent_signal_identity']=dict(wrong=True)
                torch.save(value,cp)
                with pytest.raises(ValueError):t.load_trained_model(c,'TARGET_JOINT','cpu')
                cp.write_bytes(saved)


def test_illegal_fields_and_failed_smoke_do_not_open_source_or_U(tmp_path,monkeypatch,subtests):
    with fixture(tmp_path,monkeypatch) as (t,c,pm,b,events,manifest):
        for delta in (dict(truth='private'),dict(unknown_count=4),dict(steps=1)):
            with subtests.test(delta=delta),pytest.raises(ValueError):t.validate_config(dict(c,**delta))
        monkeypatch.setattr(t,'_gradient_smoke',lambda *a:(_ for _ in ()).throw(ValueError('joint probe fail')))
        with pytest.raises(ValueError,match='joint probe fail'):t.train(c,'cpu')
        assert events==['checkpoint'] and not Path(c['output_root']).exists()
