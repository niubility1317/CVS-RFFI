"""Zero-forward flow checks: capacity, source failure, all-seven truth gate."""
import json,tempfile,types,os
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import patch
import numpy as np
from . import design as d,dispatch,report,evaluate,observer,preflight
from experiments.cvs_equivariant_identity import precision

def run():
 for row in d.rows():
  a=d.make_args(d.config(row),'cpu')
  assert (a.epochs,a.label_epochs,a.pseudo_epochs,a.lr,a.weight_decay)==(20,0,20,.0002,.0001) and not a.from_scratch and not a.a1_ema_startup_average and not a.teacher_ckpt
 # Source readout failure cannot prevent this same frozen row's predictions.
 qualifier=types.ModuleType('experiments.cvs_satellite_budget62.qualification')
 def fail(_):raise RuntimeError('toy source-readout failure')
 qualifier.run=fail
 with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as tmp:
  with patch.object(d,'BASE',Path(tmp)),patch.object(dispatch,'lock',side_effect=lambda *a,**kw:nullcontext()),patch.object(dispatch,'capacity',return_value={0:dict(pids=set(),free_mb=20000)}),patch.dict(os.environ,{'CUDA_VISIBLE_DEVICES':'0'}),patch.object(observer,'train'),patch.object(precision,'numerical_context',side_effect=lambda *a:nullcontext()),patch.dict(dispatch.sys.modules,{qualifier.__name__:qualifier}),patch.object(dispatch,'execute') as execute:
   dispatch.worker(d.rows()[0]['row_id'])
  execute.assert_called_once_with('predict',d.rows()[0]['row_id'])
 # Missing source peer is declared PARTIAL only after both workers are terminal;
 # the dispatcher then collects formal outputs and executes the zero-NN analysis.
 with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as tmp:
  base=Path(tmp)
  with patch.object(d,'BASE',base),patch.object(report,'source_decision',side_effect=ValueError('toy malformed source')) as decide:
   d.write(base/d.rows()[0]['row_id']/'source_qualification/complete.json',{})
   dispatch.decide_source();assert not (base/'source_matrix_decision.json').exists() and decide.call_count==0
   d.write(base/d.rows()[1]['row_id']/'source_qualification/complete.json',{})
   dispatch.decide_source();assert decide.call_count==1
   old=(base/'source_matrix_decision.json').read_bytes();q=d.read(base/'source_matrix_decision.json')
   assert q['status']=='SOURCE_BUDGET_READOUT_PARTIAL' and q['error'] and q['NN']==0 and q['target_feedback'] is False
   dispatch.decide_source(terminal=True);assert decide.call_count==1 and old==(base/'source_matrix_decision.json').read_bytes()
 actions=[];launched=[]
 with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as tmp:
  root=Path(tmp);base=root/'new'
  class Child:
   def __init__(self,command,**kw):self.row=command[-1];self.pid=60000+len(launched);launched.append(self.row)
   def poll(self):d.write(base/self.row/'complete.json',dict(status='TRAINED_AND_SCORED'));return 0
  decide=report.source_decision
  def decide_after_both(*a,**kw):
   assert len(launched)==2 and all((base/r['row_id']/'complete.json').is_file() for r in d.rows())
   actions.append('source');return decide(*a,**kw)
  def collect():
   assert d.read(base/'source_matrix_decision.json')['status']=='SOURCE_BUDGET_READOUT_PARTIAL'
   actions.append('formal');return dict(status='SCORED_COMPLETE_PENDING_FULL_LOG_ANALYSIS')
  with patch.object(d,'BASE',base),patch.object(d,'PROJECT',str(root)),patch.object(dispatch,'lock',side_effect=lambda *a,**kw:nullcontext()),patch.object(dispatch,'capacity',return_value={0:dict(pids={99},free_mb=20000)}),patch.object(dispatch.subprocess,'Popen',side_effect=Child),patch.object(dispatch.time,'sleep'),patch.object(preflight,'smoke'),patch.object(report,'source_decision',side_effect=decide_after_both),patch.object(report,'collect',side_effect=collect),patch.object(dispatch.os,'execve') as execute:
   dispatch.dispatch()
  assert actions==['source','formal'] and launched==[r['row_id'] for r in d.rows()]
  assert d.read(base/'queue_state.json')['max_total_processes_per_GPU']==2
  execute.assert_called_once();assert execute.call_args[0][2]['CUDA_VISIBLE_DEVICES']==''
  q=d.read(base/'source_matrix_decision.json');assert q['NN']==0 and q['model_seed_count']==1 and q['target_feedback'] is False and q['source_qualified'] is False
 # Seven complete independent dummy arrays; no actual IQ, checkpoint or truth.
 with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as tmp:
  base=Path(tmp);row=d.rows()[0]['row_id'];out=base/row/'prediction';out.mkdir(parents=True);c=d.config(d.rows()[0]);ids=np.array(['toy0','toy1']);values=np.array([0,0],np.int64);logits=np.zeros((2,6),np.float32)
  done=dict(status='PREDICTIONS_COMPLETE',config=c,count=2,views=list(d.VIEWS),truth_read=False,query_fit=False,all_parameters_frozen=True,head_fit_updates=0)
  d.write(out/'complete.json',done);np.savez_compressed(out/'predictions.npz',ids=ids,**{v:values for v in d.VIEWS})
  for view in d.VIEWS:np.savez_compressed(out/(view+'_scores.npz'),ids=ids,logits=logits)
  with patch.object(d,'BASE',base),patch.object(d,'config',return_value=c),patch.object(evaluate,'COUNT',2),patch.object(evaluate,'observations',return_value=(ids,{})):
   evaluate.preflight(row)
   done['views']=list(d.VIEWS)[:-1];d.write(out/'complete.json',done)
   try:evaluate.preflight(row)
   except ValueError:pass
   else:raise AssertionError('Missing one fixed practical view admitted to truth')
   done['views']=list(d.VIEWS);done['truth_read']=True;d.write(out/'complete.json',done)
   try:evaluate.preflight(row)
   except ValueError:pass
   else:raise AssertionError('Prediction with prior truth read admitted')
 return dict(status='PASS',flow_checks=4,CVS_NN=0,IQ=0,CP=0,target_access=False,optimizer_updates=0,source_failure_still_all7=True,terminal_only_partial=True,decision_before_formal=True,all7_truth_gate=True,cap2_actual=True)
if __name__=='__main__':print(json.dumps(run()))
