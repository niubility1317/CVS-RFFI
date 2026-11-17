import json
from pathlib import Path
from unittest.mock import patch
import pytest
from experiments.pamf_repl_priority_takeover import runner


def put(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value),encoding='utf-8')


def test_queued_job_with_existing_log_is_never_launched(tmp_path):
    root=tmp_path/'old';release=tmp_path/'release';release.mkdir()
    rt={'run_root':str(root),'release_root':str(release)}
    runtime=tmp_path/'runtime.json';put(runtime,rt)
    jid='train__candidate';jobs=[{'id':jid,'config':str(root/'config.json'),'kind':'train'}]
    put(root/'jobs.json',jobs)
    put(root/'pipeline_state.json',{'pid':99999999,'status':'RUNNING','smoke':{'status':'PASS'},
                                   'jobs':{jid:{'status':'QUEUED'}}})
    path=root/'logs'/(jid+'.log');path.parent.mkdir();path.write_text('existing')
    with pytest.raises(ValueError,match='already has a log'):
        runner.dispatch({'original_runtime_file':str(runtime),'run_root':str(tmp_path/'new')})
    assert json.loads((root/'pipeline_state.json').read_text())['pid']==99999999
    assert not (tmp_path/'new').exists()


def test_adopted_process_waits_until_exit():
    with patch.object(runner,'live',side_effect=[True,False]):
        p=runner.Adopted(123)
        assert p.poll() is None
        assert p.poll()==0 and p.returncode==0


def test_takeover_adopts_existing_and_dispatches_each_pending_once(tmp_path):
    from experiments.cric_dura_comparison import pipeline
    from experiments.pamf_physics_branch import runner as physical
    from experiments.pamf_physics_route import runner as route
    from experiments.pamf_repl_component_guard import runner as base
    from experiments.pamf_repl_completion import runner as completion
    root=tmp_path/'old';release=tmp_path/'release';release.mkdir()
    old={'run_root':str(root),'release_root':str(release)}
    runtime=tmp_path/'runtime.json';put(runtime,old)
    ids=['train__already_running','train__new_one','train__new_two']
    jobs=[{'id':key,'config':str(root/(key+'.json')),'kind':'train'} for key in ids]
    put(root/'jobs.json',jobs)
    put(root/'pipeline_state.json',{'pid':99999999,'status':'RUNNING','smoke':{'status':'PASS'},
        'admission_mib':8503,'jobs':{ids[0]:{'status':'RUNNING','pid':123,'gpu':5},
                                   ids[1]:{'status':'QUEUED'},ids[2]:{'status':'QUEUED'}}})
    launched=[]
    class Finished:
        def __init__(self,pid):self.pid=pid;self.returncode=0
        def poll(self):return 0
    def launch(rt,job,gpu,*,module):
        launched.append((job['id'],gpu,module))
        p=Finished(1000+len(launched));return p,{'status':'RUNNING','pid':p.pid,'gpu':gpu}
    rt={'original_runtime_file':str(runtime),'run_root':str(tmp_path/'new'),
        'release_root':str(tmp_path/'new-release'),'launch_owner':'test','code_commit':'commit'}
    with patch.object(runner,'live',return_value=False),patch.object(runner,'verify_worker',return_value=True),\
         patch.object(pipeline,'done',return_value=True),patch.object(pipeline,'ready',return_value=True),\
         patch.object(pipeline,'block_failed_dependencies'),patch.object(physical,'admits',return_value=True),\
         patch.object(route,'launch',side_effect=launch),patch.object(base,'analyze'),\
         patch.object(completion,'foreign_CPU_score',return_value=False),patch.object(runner.time,'sleep'):
        runner.dispatch(rt)
    assert [v[0] for v in launched]==ids[1:]
    assert [v[1] for v in launched]==[0,1]
    result=json.loads((root/'pipeline_state.json').read_text())
    assert result['status']=='ANALYZED' and all(j['status']=='COMPLETE' for j in result['jobs'].values())
    assert result['jobs'][ids[0]]['adopted_exit_status']=='unavailable'
