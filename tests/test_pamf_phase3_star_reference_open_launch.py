import copy
import json
import pytest
from experiments.pamf_phase3_open_world import star_reference_open_launch as l
from tests.test_pamf_phase3_star_reference_open_train import config as train_config
from tests.test_pamf_phase3_star_reference_open_eval import config as eval_config


def runtime(root):
    cache=root.parent/l.CACHE_RUN/'cache';rows={}
    for arm in ('C0','C1'):
        t=train_config(arm);out=str(root/'rows'/arm/'train');t.update(L_root=str(cache/'L'),OE_root=str(cache/'OE'),output_root=out);t['native_config']['output_root']=out
        e=eval_config(root,arm);e.update(train_config=t,checkpoint=out+'/final.pt',cache_root=str(cache),output_root=str(root/'rows'/arm/'predict'))
        rows[arm]=dict(train_config=t,eval_config=e)
    return dict(run_id=l.RUN,code_commit='a'*40,run_root=str(root),release_root=str(root.parent/'release'),python='python',launch_owner='one-owner',
        allowed_gpus=list(range(8)),minimum_free_mb=8192,max_added_workers_per_gpu=2,max_concurrent_rows=2,rows=rows,novel_truth='source-truth.npz',target_truth='target-truth.npz')


def test_runtime_two_owned_rows_and_no_relaunch(tmp_path):
    r=runtime(tmp_path/'run');assert l.validate(r) is r and l.PRIMARY=='C1::relation_max'
    bad=copy.deepcopy(r);bad['max_concurrent_rows']=3
    with pytest.raises(ValueError):l.validate(bad)
    (tmp_path/'run').mkdir();f=tmp_path/'runtime.json';f.write_text(json.dumps(r))
    with pytest.raises(FileExistsError):l.start(f)


@pytest.mark.parametrize('arm',['C0','C1'])
def test_real_new_stage_contracts_and_cpu_truth_isolation(monkeypatch,tmp_path,arm):
    root=tmp_path/'run';(root/'rows'/arm).mkdir(parents=True);(root/'logs').mkdir();r=runtime(root);f=tmp_path/'runtime.json';f.write_text(json.dumps(r));calls=[]
    monkeypatch.setattr(l,'_gpu_free',lambda r:[4])
    def job(r,state,path,name,cmd,gpu):
        stage=name.split('_')[-1];calls.append(stage);folder={'train':'train','eval':'predict','score':'score'}[stage];p=root/'rows'/arm/folder;p.mkdir()
        m=dict(run_id=l.RUN,code_commit=r['code_commit'],arm=arm)
        if stage=='train':
            m.update(status=l.TRAIN_STATUS,updates=2220,cumulative_updates=13320,target_reads=0,validation_reads=0,novel_validation_reads=0,original_U_s_reads=0,identity_head_replaced=False)
        elif stage=='eval':
            m.update(status=l.EVAL_STATUS,known_count=27000,novel_count=3150,target_count=864,source_truth_reads=0,target_truth_reads=0,target_Q_fit=False,formal_target_accessed=False,thresholds_frozen_before_novel_and_target=True,source_unknown_success_target_entry_gate=False)
        else:
            assert (root/'rows'/arm/'predict/complete.json').exists() and gpu is None
            assert cmd[cmd.index('--target-truth')+1]==r['target_truth'] and l._env(r)['CUDA_VISIBLE_DEVICES']==''
            m.update(status=l.SCORE_STATUS,eligible_primary=False,next_step='NO_PROMOTION',truth_last=True,formal_target_accessed=False,final_phase3_complete=False)
        if stage!='score':assert '--target-truth' not in cmd and '--novel-truth' not in cmd and gpu==4
        (p/'complete.json').write_text(json.dumps(m));return True
    monkeypatch.setattr(l,'_job',job);l.row_driver(f,arm,4)
    state=json.loads((root/'rows'/arm/'state.json').read_text());assert calls==['train','eval','score'] and state['status']=='ANALYZED_DEVELOPMENT'
