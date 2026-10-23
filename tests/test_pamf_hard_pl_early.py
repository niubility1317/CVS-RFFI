from contextlib import contextmanager
from copy import deepcopy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
from unittest.mock import patch
import pytest
from experiments.pamf_ssl_research.hard_pl_early import RUN,METHODS,configuration,validate,slot_ready
from experiments.pamf_ssl_research.hard_pl_early_publish import prepare
from experiments.pamf_ssl_research.hard_pl_early_train import aggregate,train
from experiments.pamf_ssl_research.hard_pl_mixstyle import STYLE
from tests.test_pamf_hard_pl_release import fixture
from tests.test_pamf_ssl_goal_counts import save


def state(epochs,mechanisms):
    calls=epochs*222;seen=epochs*56700;accepted=seen-100;factor=6 if mechanisms else 1
    v=dict(method='HPLT1U' if mechanisms else 'HPL2',calls=calls,seen=seen,accepted=accepted,
        student_views=factor*seen,physical_seen=0,physical_student_views=0,raw_As_seen=seen//2 if mechanisms else 0,
        contrast_anchors={name:2*accepted if mechanisms else 0 for name in ('time_down','t2.pool')},
        label_smoothing=.01,EMA_updates=0,U_truth_read=False)
    if mechanisms:v['U_style']=dict(recipe=deepcopy(STYLE),calls=6*calls,seen=6*seen,eligible=6*accepted,
        applied={name:3*accepted for name in ('time_down','t2.pool')},persistent_hooks=False,teacher_or_L_style_calls=0,
        partner_TX_truth_reads=0,shared_six_forward_plan=True,plan_calls=calls,contrast_captures_after_style=True)
    return v


@pytest.mark.parametrize('method',METHODS)
def test_earlier_schedule_changes_parent_epoch_and_budget_but_not_recipe(method):
    c=configuration(fixture()['2026092701']['F21']['train'],method,'/p','new')
    assert c['hard_pl']['start_epoch']==21 and c['hard_pl_parent_epoch']==20 and c['hard_pl_continuation_epochs']==30
    assert c['epochs']==50 and c['optimizer_updates']==11100 and c['hard_pl']['threshold']==.95 and c['hard_pl']['weight']==2
    for change in ({'hard_pl_parent_epoch':30},{'hard_pl_continuation_epochs':20},{'target_inputs':'x'},
                   {'posthoc_head_fit':True},{'hard_pl':dict(c['hard_pl'],start_epoch=31)}):
        bad=deepcopy(c);bad.update(change)
        with pytest.raises(ValueError):validate(bad)


@pytest.mark.parametrize('mechanisms',[False,True])
def test_two_new_native_segments_close_exact_total_exposures(mechanisms):
    with tempfile.TemporaryDirectory() as tmp:
        parent=Path(tmp)/'old20';save(parent/'progress.json',dict(PL_seen=396900,PL_accepted=390000))
        actual=aggregate([state(10,mechanisms),state(20,mechanisms)],[dict(folder=str(parent))])
        assert actual['hard_PL_calls']==6660 and actual['hard_PL_tail']['seen']==1701000
        assert actual['U_seen']==actual['teacher_eval_U_views']==2097900
        assert actual['student_U_views']==(10602900 if mechanisms else 2097900)
        assert actual['raw_U_As_seen']==(850500 if mechanisms else 0)
        assert actual['new_mechanism_epochs']==30 and actual['source_parent_epoch']==20
        if mechanisms:
            assert actual['U_style']['calls']==39960 and actual['U_style']['seen']==10206000
            assert actual['U_style']['plan_calls']==6660
        else:assert 'U_style' not in actual


@pytest.mark.parametrize('method',METHODS)
def test_training_uses_only_original20_then_own30_then_own50(method):
    mechanisms=method==METHODS[1]
    with tempfile.TemporaryDirectory() as tmp:
        root=Path(tmp);parent=root/'original20';save(parent/'progress.json',dict(PL_seen=396900,PL_accepted=390000))
        c=configuration(fixture()['2026092701']['F21']['train'],method,'/p','new');c['output_root']=str(root/'new-source')
        parents=[dict(folder=str(parent),seconds=5.,config=dict(parent_source=None))];seen=[]
        @contextmanager
        def installed(cfg,holder,**kwargs):
            holder['objective']=SimpleNamespace(capture_active=False,state_dict=lambda:state(cfg['segment_epochs'],mechanisms))
            yield
        def segment(cfg,device):
            seen.append(deepcopy(cfg));folder=Path(cfg['output_root']);folder.mkdir()
            save(folder/'complete.json',dict(status='SOURCE_FROZEN',epoch=cfg['segment_epochs'],updates=cfg['segment_epochs']*222,
                config=cfg,seconds=1.,source_contract=c['source_contract'],target_access=False,U_truth_read=False,head_fit_updates=0))
            (folder/'final_ssdg.pth').write_bytes(b'fixed-final-default-joint-head')
        with patch('experiments.pamf_ssl_research.hard_pl_early_train.source_parents',return_value=(parents,{'same':'contract'})), \
             patch('experiments.pamf_ssl_research.hard_pl_train.installed',side_effect=installed), \
             patch('experiments.cric_dura_comparison.f_train.train',side_effect=segment):train(c,'cpu')
        assert [(v['segment_epochs'],v['epoch_offset'],v['step_offset']) for v in seen]==[(10,20,4440),(20,30,6660)]
        assert seen[0]['parent_source']==str(parent) and seen[1]['parent_source']==str(root/'new-source/segment30')
        done=json.loads((root/'new-source/complete.json').read_text())
        assert done['epoch']==50 and done['updates']==11100 and done['incremental_epochs']==30 and done['incremental_updates']==6660
        assert len(done['segments'])==3 and done['segments'][0]['folder']==str(parent)
        assert (root/'new-source/final_ssdg.pth').read_bytes()==b'fixed-final-default-joint-head'


def test_parent_protocol_rejects_target_or_wrong_E20_before_checkpoint_load():
    from experiments.pamf_ssl_research.hard_pl_early_train import source_parents
    c=configuration(fixture()['2026092701']['F21']['train'],METHODS[0],'/p','new')
    with patch('experiments.pamf_ssl_research.hard_pl_early_train.read',side_effect=[{},dict(status='BAD',config={}),{}]), \
         patch('experiments.pamf_ssl_research.hard_pl_early_train.torch.load') as load:
        with pytest.raises(ValueError,match='DATA_CONTRACT'):source_parents(c)
        load.assert_not_called()


def test_six_rows_84_target_cells_and_complete_matching_source_configs():
    rt,blobs=prepare(fixture(),'/p','/p/releases/'+RUN,'new')
    assert rt['methods']==list(METHODS) and len(rt['rows'])==6 and rt['max_total_workers']==6
    assert rt['prior_score_owner_state']==rt['NAMS_GPU_owner_state']
    for row in rt['rows']:
        stem='configs/'+row['row_id'];c,p,s=(blobs[stem+'-'+k+'.json'] for k in ('train','predict','score'))
        assert c['hard_pl_parent_epoch']==20 and c['hard_pl_continuation_epochs']==30
        assert list(p['readouts'])==s['readouts']==['raw','DURA-LU'] and 'truth' not in p['tests']['main']
        assert p['source_output']==c['output_root']


def test_each_arm_waits_its_own_predecessor_and_reserves_starting_peer_PID():
    from experiments.pamf_ssl_research.control import pending_gpu_pids,capacity
    seed=2026092701;rt=dict(gpu_by_seed={str(seed):5},gpu_predecessor_state='m',NAMS_GPU_owner_state='n')
    states=dict(m=dict(jobs={'HPL_MS2-s'+str(seed):dict(status='RUNNING',pid=89)}),
                n=dict(jobs={'HPL_NAMS2-s'+str(seed):dict(status='RUNNING',pid=99)}));jobs={}
    with patch('experiments.cric_dura_comparison.io.read',side_effect=lambda p:states[p]):
        assert not slot_ready(rt,seed,METHODS[0],jobs)
        states['m']['jobs']['HPL_MS2-s'+str(seed)]['status']='SCORED'
        assert slot_ready(rt,seed,METHODS[0],jobs) and not slot_ready(rt,seed,METHODS[1],jobs)
        jobs[METHODS[0]+'-s'+str(seed)]=dict(status='RUNNING',pid=101)
        active=[dict(gpu=5,child=SimpleNamespace(pid=101))]
        assert pending_gpu_pids(rt,METHODS,seed,active)=={99,101}
        with patch('experiments.pamf_ssl_research.control.subprocess.check_output',side_effect=['12\n13\n','19000']):
            assert capacity(5,pending_gpu_pids(rt,METHODS,seed))==(3,19000)
        states['n']['jobs']['HPL_NAMS2-s'+str(seed)]['status']='SCORED'
        assert slot_ready(rt,seed,METHODS[1],jobs)


def test_real_E20_smoke_first_then_no_formal_launch_until_PASS():
    from experiments.pamf_ssl_research.control import run
    class ReachedSmoke(Exception):pass
    with tempfile.TemporaryDirectory() as tmp:
        rt,_=prepare(fixture(),'/p',tmp,'new');rt.update(run_root=tmp+'/output',runtime_file=tmp+'/runtime.json')
        def reached(command,**kwargs):
            assert 'experiments.pamf_ssl_research.hard_pl_early_worker' in command
            raise ReachedSmoke
        with patch('experiments.pamf_ssl_research.hard_pl_early.slot_ready',return_value=True), \
             patch('experiments.pamf_ssl_research.hard_pl_early.reserved_predecessor_pids',return_value=(99,)), \
             patch('experiments.pamf_ssl_research.control.capacity',return_value=(2,19000)), \
             patch('experiments.pamf_ssl_research.control.subprocess.run',side_effect=reached), \
             patch('experiments.pamf_ssl_research.control.subprocess.Popen') as launch:
            with pytest.raises(ReachedSmoke):run(rt)
            launch.assert_not_called()


def test_six_model_both_readout_report_preserves_same_raw_thresholds():
    from experiments.pamf_ssl_research.dura_reference_report import report
    from tests.test_pamf_ssl_dura_reference import report_fixture
    with tempfile.TemporaryDirectory() as tmp:
        root=Path(tmp);rt=report_fixture(root);rt.update(run_id=RUN,methods=list(METHODS))
        rt['rows']=[r for r in rt['rows'] if r['method'] in ('PSM','RXPSM')]
        for row in rt['rows']:
            method=METHODS[0] if row['method']=='PSM' else METHODS[1]
            row.update(method=method,row_id=method+'-s'+str(row['model_seed']))
            c=json.loads(Path(row['score_config']).read_text());c.update(run_id=RUN,row_id=row['row_id']);save(Path(row['score_config']),c)
            path=Path(c['output_root'])/'scores.json';v=json.loads(path.read_text());v.update(run_id=RUN,row_id=row['row_id']);save(path,v)
        report(rt,root/'report');a=json.loads((root/'report/analysis.json').read_text())
        assert {v['method'] for v in a['verdicts']}==set(METHODS) and len(a['verdicts'])==4 and not a['goal_met']
