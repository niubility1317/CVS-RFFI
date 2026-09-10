import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import pytest

spec=importlib.util.spec_from_file_location('handoff',Path(__file__).resolve().parents[1]/'code/scripts/reassign_a1_pending_gpu.py')
h=importlib.util.module_from_spec(spec);spec.loader.exec_module(h)


def test_only_pending_gpu_changes():
    matrix={'rows':[{'id':'X2','gpu':6,'options':{'--epochs':'400'}},{'id':h.ROW,'gpu':0,'options':{'--epochs':'400'}}]}
    original=copy.deepcopy(matrix)
    state={'waiting_rows':[h.ROW],'rows':{'X2':{'status':'RUNNING'}}}
    changed=h.changed_matrix(matrix,state,7)
    assert matrix==original and changed['rows'][0]==original['rows'][0]
    assert changed['rows'][1]==dict(original['rows'][1],gpu=7)
    state['rows'][h.ROW]={'status':'RUNNING'}
    with pytest.raises(RuntimeError):h.changed_matrix(matrix,state,7)


@pytest.mark.parametrize('ready',[True,False])
def test_dispatcher_transfer_never_signals_training_workers(tmp_path,monkeypatch,ready):
    # Windows test host simulates the Linux signal constants; os.kill is mocked below.
    monkeypatch.setattr(h,'signal',SimpleNamespace(SIGSTOP=19,SIGTERM=15,SIGCONT=18))
    monkeypatch.setattr(h,'PROJECT',tmp_path);monkeypatch.setattr(h,'RELEASE',tmp_path/'release')
    root=tmp_path/'runs'/h.RUN;root.mkdir(parents=True)
    (tmp_path/'logs'/h.RUN).mkdir(parents=True)
    state={'pid':100,'waiting_rows':[h.ROW],'rows':{'X2':{'pid':101,'gpu':6,'status':'RUNNING'}}}
    matrix={'core90_options':{},'rows':[{'id':'X2','gpu':6,'options':{}},{'id':h.ROW,'gpu':0,'options':{}}]}
    (root/'pipeline_state.json').write_text(json.dumps(state));(root/'effective_matrix.json').write_text(json.dumps(matrix))
    monkeypatch.setattr(h,'load_runner',lambda:SimpleNamespace(gpu_compute_pids=lambda gpu:{'555'}))
    current={'status':'R','exists':True}; signals=[];terminated=[]
    def info(pid):
        if pid==100:
            if not current['exists']:return None
            return {'status':current['status'],'cwd':str(h.RELEASE),'argv':['python',str(h.RELEASE/'code/scripts/run_a1_extended_budgets.py'),'--run-id',h.RUN]}
        assert pid==101
        return {'status':'R','cwd':str(h.RELEASE),'argv':['python','train.py','--output_dir',str(root/'X2')]}
    monkeypatch.setattr(h,'process_info',info)
    def kill(pid,sig):
        signals.append((pid,sig));assert pid==100
        if sig==h.signal.SIGSTOP:current['status']='T'
        if sig==h.signal.SIGTERM:current['exists']=False
        if sig==h.signal.SIGCONT:current['status']='R'
    monkeypatch.setattr(h.os,'kill',kill)
    def popen(argv,**kw):
        assert '--manage' in argv and kw['start_new_session']
        if ready:Path(argv[-1]).with_suffix('.ready').write_text('999')
        return SimpleNamespace(pid=999,poll=lambda:None if ready else 1,terminate=lambda:terminated.append(True))
    monkeypatch.setattr(h.subprocess,'Popen',popen)
    if ready:
        h.handoff(7)
        assert (root/'gpu_reassignment_20260910_r1.activate').exists()
        assert not current['exists']
        assert h.read(root/'gpu_reassignment_20260910_r1.json')['matrix']['rows'][1]['gpu']==7
    else:
        with pytest.raises(RuntimeError,match='not ready'):h.handoff(7)
        assert current=={'exists':True,'status':'R'}
        assert not (root/'gpu_reassignment_20260910_r1.activate').exists()
        assert not any(sig==h.signal.SIGTERM for _,sig in signals)
    assert not (root/h.ROW).exists()  # Transfer itself never launches the training row.
    assert h.read(root/'pipeline_state.json')==state
