"""One derived-source response fit; no raw IQ, checkpoint or training access."""
import os,sys,time
from pathlib import Path
import numpy as np
from . import design as d
from .core import evaluate
from .permissions import validate
def run():
 if len(sys.argv)!=1 or os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('CPU-only fixed source entry required')
 c=d.config();d.validate(c);project=d.read(d.ROOT/'runtime.json')['project'];out=d.resolve(c['output_root'],project);out.mkdir(parents=True,exist_ok=False);start=time.perf_counter();d.write(out/'active.json',dict(pid=os.getpid(),cwd=os.getcwd(),argv=sys.argv,owner=d.OWNER))
 try:
  from .checks import run as checks
  d.write(out/'checks.json',checks());print('SMOKE PASS complete source handles and response controls;0NN/CP/rawIQ',flush=True)
  source=d.resolve(d.DERIVED,project);done=d.read(source/'completion.json');resolved=d.read(source/'resolved_config.json');validate(done,resolved)
  with np.load(source/'cell_matrices.npz',allow_pickle=False) as a:
   if set(a.files)!={'L_s','V'}:raise ValueError('Only fixed original L/V matrices required')
   L,V=a['L_s'].copy(),a['V'].copy()
  result,hs=evaluate(L,V);result.update(status='SOURCE_SHARED_RESPONSE_READOUT_COMPLETE',run_id=d.RUN,source_L=6300,source_V=27000,all_source_RX=list(d.RX),response_fits=72,predicted_V_cells=90,raw_IQ_reads=0,CP=0,NN=0,updates=0,U_truth_read=False,automatic_training=False)
  d.write(out/'resolved_config.json',dict(config=c,execution_commit=(d.ROOT/'release_commit.txt').read_text().strip(),source_handle_exact_match=True,derived_source_commit=resolved['execution_commit'],source_RX_holdout=None,model_constructed=False,raw_IQ_read=False,target_access=False));np.savez_compressed(out/'responses.npz',responses=hs);d.write(out/'result.json',result)
  d.write(out/'completion.json',dict(status='ARTIFACTS_COMPLETE_SOURCE_ONLY',run_id=d.RUN,decision=result['decision'],source_L=6300,source_V=27000,source_RX=list(d.RX),source_RX_holdout=None,source_cell_matrices_read=180,source_matrix_entries=72000,closed_form_response_fits=72,registered_folds=18,predicted_V_cells=90,saved_complex_response_entries=1800,raw_IQ_reads=0,CP=0,NN=0,teacher_NN=0,updates=0,head_fit_updates=0,U_used=False,U_truth_read=False,target_access=False,IQ_views_generated=0,automatic_training=False,joint_goal_complete=False,wall_seconds=time.perf_counter()-start));print('SOURCE_RESPONSE_COMPLETE '+result['decision'],flush=True)
 except Exception as exc:
  d.write(out/'failure.json',dict(status='FAILED',error=repr(exc),automatic_retry=False));raise
if __name__=='__main__':run()
