"""One immutable source-only no-refit normalization-order readout."""
import os,sys,time
import numpy as np
from . import design as d
from .core import evaluate
from .permissions import validate
def run():
 if len(sys.argv)!=1 or os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('Fixed CPU source-only entry required')
 c=d.config();d.validate(c);project=d.read(d.ROOT/'runtime.json')['project'];out=d.resolve(c['output_root'],project);out.mkdir(parents=True,exist_ok=False);start=time.perf_counter();d.write(out/'active.json',dict(pid=os.getpid(),cwd=os.getcwd(),argv=sys.argv,owner=d.OWNER))
 try:
  from .checks import run as checks
  d.write(out/'checks.json',checks());print('SMOKE PASS no-refit normalized-packet controls;0CP/NN/rawIQ',flush=True)
  source=d.resolve(d.DERIVED,project);response=d.resolve(d.RESPONSE,project);sd=d.read(source/'completion.json');sr=d.read(source/'resolved_config.json');rd=d.read(response/'completion.json');rr=d.read(response/'resolved_config.json');validate(sd,sr,rd,rr);old=d.read(response/'result.json')
  with np.load(source/'cell_matrices.npz',allow_pickle=False) as a:L,V=a['L_s'].copy(),a['V'].copy()
  with np.load(response/'responses.npz',allow_pickle=False) as a:
   if set(a.files)!={'responses'}:raise ValueError('Only fixed original responses required')
   H=a['responses'].copy()
  with np.load(source/'observations.npz',allow_pickle=False) as a:u=a['V_factor_real']+1j*a['V_factor_imag']
  vmeta=d.read(source/'source_ids.json')['V'];meta={k:np.asarray(vmeta[k]) for k in ('rx','day','y')};del vmeta
  result,pred=evaluate(L,V,H,u,meta,old);result.update(status='SOURCE_PACKETWISE_ORDER_COMPLETE',run_id=d.RUN)
  d.write(out/'resolved_config.json',dict(config=c,execution_commit=(d.ROOT/'release_commit.txt').read_text().strip(),original_source_commit=sr['execution_commit'],fixed_response_commit=rr['execution_commit'],input_handles_exact_match=True,source_RX_holdout=None,raw_IQ_reads=0,model_constructed=False,target_access=False));np.savez_compressed(out/'predicted_matrices.npz',centered_predictions=pred);d.write(out/'result.json',result)
  d.write(out/'completion.json',dict(status='ARTIFACTS_COMPLETE_SOURCE_ONLY',run_id=d.RUN,decision=result['decision'],source_L=6300,source_V=27000,source_RX=list(d.RX),source_RX_holdout=None,source_matrix_entries_read=72000,V_factor_vectors_read=27000,V_complex_factor_entries_read=540000,V_reference_vectors_used=5400,fixed_complex_responses_read=1800,registered_folds=18,predicted_V_cells=90,packetwise_response_vector_transforms=27000,saved_complex_prediction_entries=36000,response_refits=0,raw_IQ_reads=0,CP=0,NN=0,teacher_NN=0,updates=0,head_fit_updates=0,U_used=False,U_truth_read=False,target_access=False,IQ_views_generated=0,automatic_training=False,old_mean_proxy_decision_unchanged=True,joint_goal_complete=False,wall_seconds=time.perf_counter()-start));print('SOURCE_PACKET_ORDER_COMPLETE '+result['decision'],flush=True)
 except Exception as exc:
  d.write(out/'failure.json',dict(status='FAILED',error=repr(exc),automatic_retry=False));raise
if __name__=='__main__':run()
