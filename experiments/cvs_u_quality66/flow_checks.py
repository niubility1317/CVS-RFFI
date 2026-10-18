"""Four zero-network checks for four-row completion and independent truth gating."""
import json,os,tempfile,types
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import patch
import numpy as np
from . import design as d,dispatch,report,evaluate,observer,preflight
from experiments.cvs_equivariant_identity import precision

def run():
    a=d.make_args(d.config(d.rows()[0]),'cpu');assert (a.epochs,a.label_epochs,a.pseudo_epochs,a.lr)==(10,0,10,.0002) and not a.from_scratch and not a.teacher_ckpt
    qualifier=types.ModuleType('experiments.cvs_u_quality66.qualification')
    def fail(_):raise RuntimeError('toy source failure')
    qualifier.run=fail
    with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as tmp:
        with patch.object(d,'BASE',Path(tmp)),patch.object(dispatch,'lock',side_effect=lambda *a,**kw:nullcontext()),patch.object(dispatch,'capacity',return_value={0:dict(pids=set(),free_mb=20000)}),patch.dict(os.environ,{'CUDA_VISIBLE_DEVICES':'0'}),patch.object(observer,'train'),patch.object(precision,'numerical_context',side_effect=lambda *a:nullcontext()),patch.dict(dispatch.sys.modules,{qualifier.__name__:qualifier}),patch.object(dispatch,'execute') as execute:
            dispatch.worker(d.rows()[0]['row_id'])
        execute.assert_called_once_with('predict',d.rows()[0]['row_id'])
    # A not-yet-finished peer is not a source failure. Both qualification files
    # must be ready; a malformed complete pair becomes one retained PARTIAL.
    with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as tmp:
        base=Path(tmp)
        with patch.object(d,'BASE',base),patch.object(report,'source_decision',side_effect=ValueError('toy malformed source')) as decide:
            dispatch.decide_source();assert not (base/'source_matrix_decision.json').exists() and decide.call_count==0
            d.write(base/d.rows()[0]['row_id']/'source_qualification/complete.json',{})
            dispatch.decide_source();assert not (base/'source_matrix_decision.json').exists() and decide.call_count==0
            for row in d.rows()[1:-1]:d.write(base/row['row_id']/'source_qualification/complete.json',{})
            dispatch.decide_source();assert not (base/'source_matrix_decision.json').exists() and decide.call_count==0
            d.write(base/d.rows()[-1]['row_id']/'source_qualification/complete.json',{})
            dispatch.decide_source();assert decide.call_count==1
            old=(base/'source_matrix_decision.json').read_bytes();q=d.read(base/'source_matrix_decision.json');assert q['status']=='SOURCE_READOUT_PARTIAL' and q['error'] and not q['target_feedback'] and q['NN']==0
            dispatch.decide_source(terminal=True);assert decide.call_count==1 and old==(base/'source_matrix_decision.json').read_bytes()
    actions=[];launched=[]
    with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as tmp:
        base=Path(tmp)/'new';release=Path(tmp)/'release';release.mkdir();(release/'release_commit.txt').write_text('toy-commit',encoding='utf8')
        class Child:
            def __init__(self,command,**kw):self.row=command[-1];self.pid=60001+len(launched);self.returncode=None;launched.append(self.row)
            def poll(self):self.returncode=0;d.write(base/self.row/'complete.json',dict(status='TRAINED_AND_SCORED'));return 0
        decide=report.source_decision
        def terminal_source():
            assert len(launched)==2 and all((base/r['row_id']/'complete.json').is_file() for r in d.rows());actions.append('source');return decide()
        def collect():
            assert (base/'source_matrix_decision.json').is_file();actions.append('formal');return dict(status='FORMAL_COMPLETE_SOURCE_PARTIAL')
        states=[{0:dict(pids={98,99},free_mb=20000)},{0:dict(pids={99},free_mb=20000)}]
        def capacity():return states.pop(0) if len(states)>1 else states[0]
        with patch.object(d,'BASE',base),patch.object(d,'PROJECT',tmp),patch.object(d,'ROOT',release),patch.object(dispatch,'lock',side_effect=lambda *a,**kw:nullcontext()),patch.object(dispatch,'capacity',side_effect=capacity),patch.object(dispatch.subprocess,'Popen',side_effect=Child),patch.object(dispatch.time,'sleep'),patch.object(preflight,'smoke'),patch.object(report,'source_decision',side_effect=terminal_source),patch.object(report,'collect',side_effect=collect),patch.object(dispatch.os,'execve') as execute:
            dispatch.dispatch()
        assert actions==['source','formal'] and launched==[r['row_id'] for r in d.rows()]
        done=d.read(base/'complete.json');assert done['new_model_rows']==2 and done['reused_controls']==2
        assert d.read(base/'queue_state.json')['max_total_processes_per_GPU']==2
        execute.assert_called_once();command=execute.call_args[0][1];assert execute.call_args[0][2]['CUDA_VISIBLE_DEVICES']=='' and command[command.index('--local-nn')+1]==str(d.LOCAL_NN) and command[-2:]==['--code-commit','toy-commit']
        assert d.read(base/'analysis_launch.json')['logical_rows']==4
    # Accept seven dummy arrays, reject missing views and truth-tainted flags.
    with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as tmp:
        base=Path(tmp);row=d.rows()[0]['row_id'];out=base/row/'prediction';out.mkdir(parents=True);c=d.config(d.rows()[0]);ids=np.array(['toy0','toy1']);v=np.zeros(2,np.int64);z=np.zeros((2,6),np.float32);done=dict(status='PREDICTIONS_COMPLETE',config=c,count=2,views=list(d.VIEWS),truth_read=False,query_fit=False,all_parameters_frozen=True,head_fit_updates=0)
        d.write(out/'complete.json',done);np.savez_compressed(out/'predictions.npz',ids=ids,**{view:v for view in d.VIEWS})
        for view in d.VIEWS:np.savez_compressed(out/(view+'_scores.npz'),ids=ids,logits=z)
        with patch.object(d,'BASE',base),patch.object(d,'config',return_value=c),patch.object(evaluate,'COUNT',2),patch.object(evaluate,'observations',return_value=(ids,{})):
            evaluate.preflight(row)
            done['views']=list(d.VIEWS)[:-1];d.write(out/'complete.json',done)
            try:evaluate.preflight(row)
            except ValueError:pass
            else:raise AssertionError('Incomplete seven-view row admitted to truth')
            done['views']=list(d.VIEWS);done['truth_read']=True;d.write(out/'complete.json',done)
            try:evaluate.preflight(row)
            except ValueError:pass
            else:raise AssertionError('Truth-tainted predictor admitted')
    return dict(status='PASS',flow_checks=4,CVS_NN=0,IQ=0,CP=0,target_access=False,optimizer_updates=0,source_failure_still_all7=True,incomplete_peer_not_source_failure=True,source_partial_preserved=True,two_fixed_rows=True,decision_before_formal_comparison=True,automatic_CPU_analysis=True,cap2_actual=True,all7_truth_gate=True)
if __name__=='__main__':print(json.dumps(run()))
