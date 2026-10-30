import json
from pathlib import Path
import pytest
from experiments.pamf_phase3_open_world import star_open_feature_launch as l
from tests.test_pamf_phase3_star_open_feature_export import config as export_config


def runtime(root):
    return dict(run_id=l.RUN,code_commit='b'*40,run_root=str(root),release_root=str(root.parent/'release'),python='python',launch_owner='one-owner',
        allowed_gpus=list(range(8)),minimum_free_mb=4096,max_added_workers_per_gpu=2,export_config=export_config(root/'features'),novel_truth='only-source-truth.npz')


def test_runtime_whitelist_and_no_output_relaunch(tmp_path):
    r=runtime(tmp_path/'run');assert l.validate(r) is r
    with pytest.raises(ValueError):l.validate(dict(r,target_truth='forbidden'))
    (tmp_path/'run').mkdir();f=tmp_path/'runtime.json';f.write_text(json.dumps(r))
    with pytest.raises(FileExistsError):l.start(f)


def test_export_first_and_separate_cpu_score(monkeypatch,tmp_path):
    root=tmp_path/'run';root.mkdir();(root/'configs').mkdir();(root/'logs').mkdir();r=runtime(root)
    f=tmp_path/'runtime.json';f.write_text(json.dumps(r));calls=[];free=iter([[],[6]])
    monkeypatch.setattr(l,'_gpu_free',lambda r:next(free));monkeypatch.setattr(l.time,'sleep',lambda seconds:calls.append(('wait',seconds)))
    def job(r,state,path,name,cmd,gpu):
        calls.append((name,gpu,cmd));folder='features' if name=='export' else 'score';(root/folder).mkdir()
        marker=dict(run_id=l.RUN,code_commit=r['code_commit'],source_count=3150,target_reads=0,optimizer_updates=0)
        if name=='export':
            assert '--novel-truth' not in cmd and gpu==6
            marker.update(status='STAR_OPEN_FEATURES_COMPLETE',private_truth_reads=0,model_states_unchanged=True,source_role='novel')
        else:
            assert (root/'features/complete.json').exists() and gpu is None and cmd[cmd.index('--novel-truth')+1]==r['novel_truth']
            assert l._env(r)['CUDA_VISIBLE_DEVICES']==''
            marker.update(status='STAR_OPEN_FEATURE_SCORES_COMPLETE',truth_last=True,all_neighbors_fixed_before_truth=True,final_phase3_complete=False)
        (root/folder/'complete.json').write_text(json.dumps(marker));return True
    monkeypatch.setattr(l,'_job',job);l.driver(f)
    state=json.loads((root/'pipeline_state.json').read_text());assert state['status']=='ANALYZED_SOURCE_DIAGNOSTIC' and not state['final_phase3_complete']
    assert [v[0] for v in calls]==['wait','export','score']


def test_failed_export_never_opens_truth_scorer(monkeypatch,tmp_path):
    root=tmp_path/'run';root.mkdir();(root/'logs').mkdir();r=runtime(root);f=tmp_path/'runtime.json';f.write_text(json.dumps(r));calls=[]
    monkeypatch.setattr(l,'_gpu_free',lambda r:[1]);monkeypatch.setattr(l,'_job',lambda r,state,path,name,cmd,gpu:(calls.append(name) or False))
    l.driver(f);state=json.loads((root/'pipeline_state.json').read_text())
    assert calls==['export'] and state['status']=='FAILED_PRESERVED'
