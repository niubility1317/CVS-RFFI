"""Immutable old F runtime dependency;one CPU-only no-training probe."""
import importlib.util,json,sys,types
from pathlib import Path
def bootstrap():
 here=Path(__file__).resolve().parents[2];backend=Path(json.loads((here/'runtime.json').read_text(encoding='utf8'))['backend'])
 spec=importlib.util.spec_from_file_location('frequency_input_backend',backend/'experiments/cvs_crossrx_satmix20/runner.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);m.bootstrap()
 name='experiments.cvs_cfo_input16';package=types.ModuleType(name);package.__path__=[str(here/'experiments/cvs_cfo_input16')];sys.modules[name]=package;return here
def main():
 if len(sys.argv)!=1:raise ValueError('No frequency,seed,data or training overrides allowed')
 bootstrap()
 from experiments.cvs_cfo_input16 import design as d
 from experiments.cvs_cfo_input16.preflight import smoke
 d.BASE.mkdir(parents=True,exist_ok=False)
 try:
  d.write(d.BASE/'request.json',d.config());audit=smoke();done=dict(status='ENGINEERING_INPUT_ONLY_COMPLETE',rows=1,input_effect_decision=audit['qualification']['decision'],actual_NN_packets=162,source_IQ_reads=0,new_epochs=0,new_optimizer_steps=0,head_fit_updates=0,target_access=False,identity_accuracy_computed=False,Ag_efficacy_proved=False,joint_goal_complete=False)
  d.write(d.BASE/'completion.json',done);print(json.dumps(done),flush=True)
 except Exception as error:
  d.write(d.BASE/'failure.json',dict(status='TECHNICAL_FAILURE_NO_IDENTITY_RESULT',error=repr(error),target_access=False));raise
if __name__=='__main__':main()
