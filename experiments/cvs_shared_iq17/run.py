"""One frozen source-only common-direction question;smoke first,all12 folds."""
import os,random,sys,time
from pathlib import Path
import torch
from . import design as d
from .iq_direction import make_direction,permuted_control,complex_rms,EPS,PEAK
from .question import folds,qualify,EXPECTED
from .scoring import all_scores
from experiments.cvs_commonphase_source.run import checkpoint_model
from experiments.cvs_commonphase_source.core import select_fit
from experiments.cvs_sigtime_heldcredit.held_credit import fresh_parameter_caches

def run():
 if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('Fixed CPU-only diagnostic required')
 torch.set_num_threads(2);c=d.config();d.validate(c);out=d.BASE;out.mkdir(parents=True,exist_ok=False);started=time.perf_counter();d.write(out/'active.json',dict(pid=os.getpid(),cwd=os.getcwd(),argv=sys.argv,owner=d.OWNER))
 try:
  m,trained=checkpoint_model();before={k:v.detach().clone() for k,v in m.state_dict().items()};ref=m.id_backbone.encoder
  cache=lambda:fresh_parameter_caches(m)
  def logits(x):return m.forward_identity_only(x)['tx_logits']
  with torch.no_grad(),cache():z=logits(torch.randn(2,2,256,generator=torch.Generator().manual_seed(202610178003)))
  if z.shape!=(2,6) or not bool(torch.isfinite(z).all()) or ref._view_eval_packets!=2:raise ValueError('Real checkpoint no-query smoke failed')
  d.write(out/'real_checkpoint_smoke.json',dict(status='PASS',strict_load=True,matching_own_source_final20=True,synthetic_NN_packets=2,source_IQ_reads=0,optimizer_updates=0,target_access=False));print('SMOKE PASS matching real checkpoint;2syntheticNN;no query;continue',flush=True)
  carrier=d.read(d.SOURCE);prepared=d.read(Path(d.PREPARATION)/'preparation_complete.json')
  if prepared['status']!='SOURCE_GEOMETRY_AUDIT_COMPLETE' or prepared['target_access'] or prepared['identity_updates']!=0:raise ValueError('Existing source preparation differs')
  packed=torch.load(Path(d.PREPARATION)/'source_cache.pt',map_location='cpu',weights_only=False)
  for actual in (packed['contract'],prepared['source_contract']):
   if any(actual.get(k)!=v for k,v in carrier.items()) or actual.get('native_role_comparison')!='EXACT_MATCH':raise ValueError('Existing source role handles differ')
  L=select_fit({k:packed['labeled'][k] for k in ('x','y','rx','day','ids')},carrier,trained);del packed
  ids=L['ids'];y=L['y'].tolist();rx=L['rx'].tolist();day=L['day'].tolist();matrix=folds(ids,y,rx,day)
  d.write(out/'source_contract.json',trained);d.write(out/'fit_L_ids.json',ids);d.write(out/'folds.json',matrix)
  d.write(out/'resolved_config.json',dict(config=c,commit=(d.ROOT/'release_commit.txt').read_text().strip(),python=sys.executable,torch_version=torch.__version__,checkpoint_provenance='Own scratch20/final20/source-only/exactfit roles;matching known source reused,no new training initialization',source_cache_other_roles_and_features_deserialized_but_unused=True,fit_L_unique=5040,opaque_role_handles=True,source_IQ_transform='Only raw plus one shared bounded additive vector;no RMS rerun or signal clipping',U_truth_read=False,held_RX1_used_in_math=False,target_access=False))
  print('INPUT PASS exactfitL5040/opaqueIDs/RXday;heldRX1/U/V/target unused;frozen original head',flush=True)
  rng=torch.get_rng_state().clone();pyrng=random.getstate();pred={v:[None]*5040 for v in d.VIEWS};directions=[];reference_NN=backward_packets=backward_calls=evaluation_NN=0;identifiable=0
  (out/'predictions').mkdir()
  for fold in matrix:
   key='RX%d_day%d'%(fold['RX'],fold['day']);a=fold['reference'];b=fold['held'];clock=time.perf_counter()
   batches=(L['x'][a[start:start+128]] for start in range(0,len(a),128))
   delta,evidence=make_direction(m,batches,logits,fold['seed'],cache_context=cache);control=permuted_control(delta,fold['control_seed']);identifiable+=int(evidence['identifiable']);reference_NN+=evidence['eval_forward_packets'];backward_packets+=evidence['input_backward_packets'];backward_calls+=evidence['input_backward_calls']
   drms=float(complex_rms(delta)[0]);crms=float(complex_rms(control)[0]);peak=float(delta.abs().max());cpeak=float(control.abs().max())
   if drms>EPS*(1+1e-6) or peak>PEAK*(1+1e-6) or abs(drms-crms)>EPS*1e-6 or peak!=cpeak or float(delta.mean(-1).abs().max())>1e-9 or float(control.mean(-1).abs().max())>1e-9:raise ValueError('Fixed direction/DC/same-energy control bound differs')
   row=dict(RX=fold['RX'],day=fold['day'],reference_count=len(a),held_count=len(b),direction=delta[0].double().tolist(),permuted=control[0].double().tolist(),operator=evidence,direction_RMS=drms,control_RMS=crms,direction_peak=peak,control_peak=cpeak,reference_forward_packets=evidence['eval_forward_packets'],input_backward_packets=evidence['input_backward_packets'],input_backward_calls=evidence['input_backward_calls'],views={})
   for view,addition in (('raw',None),('direction',delta),('permuted',control)):
    decisions=[];saved_logits=[];raw_power=[];relative=[]
    for start in range(0,len(b),128):
     indices=b[start:start+128];raw=L['x'][indices];x=raw if addition is None else raw+addition
     with torch.no_grad(),cache():z=logits(x).double()
     if z.shape!=(len(indices),6) or not bool(torch.isfinite(z).all()):raise ValueError('Complete finite original default-head readout required')
     pp=raw.double().square().sum(1).mean(1);raw_power+=pp.tolist();relative+=((complex_rms(addition).expand(len(raw))/pp.sqrt()).tolist() if addition is not None else [0.]*len(raw))
     hard=z.argmax(1).tolist();decisions+=hard;saved_logits+=z.tolist();evaluation_NN+=len(indices)
     for i,p in zip(indices,hard):
      if pred[view][i] is not None:raise ValueError('Repeated held physical decision')
      pred[view][i]=p
    row['views'][view]=dict(predictions=decisions,logits=saved_logits,raw_complex_power_min=min(raw_power),raw_complex_power_max=max(raw_power),relative_added_RMS_max=max(relative))
   row.update(held_indices=b,physical_ids=[ids[i] for i in b],labels=[y[i] for i in b],seconds=time.perf_counter()-clock)
   d.write(out/'predictions'/(key+'.json'),row);directions.append(row)
   d.write(out/'progress.json',dict(completed_folds=len(directions),total_folds=12,reference_NN=reference_NN,input_backward_packets=backward_packets,input_backward_calls=backward_calls,source_eval_NN=evaluation_NN,total_NN_so_far=2+reference_NN+evaluation_NN,optimizer_updates=0,target_access=False))
   print('FOLD '+key+' complete;referenceNN='+str(reference_NN)+' inputgrad='+str(backward_packets)+' sourceevalNN='+str(evaluation_NN),flush=True)
  if any(v is None for p in pred.values() for v in p):raise ValueError('All5040 decisions perview required')
  if m.training or any(p.requires_grad or p.grad is not None for p in m.parameters()) or any(not torch.equal(v,m.state_dict()[k]) for k,v in before.items()) or not torch.equal(rng,torch.get_rng_state()) or pyrng!=random.getstate() or ref._view_train_packets or m._joint_calls or m._joint_As_packets or m._supcon_calls or m._supcon_pending is not None or m._supcon_handles:raise ValueError('Frozen state/globalRNG/unused mechanisms changed')
  execution=dict(EXPECTED);execution.update(identifiable_folds=identifiable,reference_forward_packets=reference_NN,input_backward_packets=backward_packets,input_backward_calls=backward_calls,source_eval_packets=evaluation_NN,total_forward_packets=ref._view_eval_packets,target_access=False)
  if any(execution[k]!=v for k,v in EXPECTED.items()):raise ValueError('Actual source cost differs from registered budget')
  scores=all_scores(y,pred,rx,day);decision=qualify(scores,execution)
  result=dict(status='ANALYZED_SOURCE_ONLY',scores=scores,qualification=decision,execution=execution,physical_L_packets=5040,source_decisions=15120,synthetic_smoke_NN_packets=2,actual_NN_packets=ref._view_eval_packets,actual_input_backward_packets=backward_packets,actual_input_backward_calls=backward_calls,actual_optimizer_updates=0,head_fit_updates=0,default_head_only=True,model_state_unchanged=True,NoExplicitCFO=True,teacher_forwards=0,new_training_loss_terms=0,As_generations=0,U_used=False,V_used=False,held_RX1_used=False,target_access=False,elapsed_seconds=time.perf_counter()-started)
  d.write(out/'result.json',result);d.write(out/'completion.json',dict(status='ANALYZED_SOURCE_ONLY',scientific_decision='PASS_SENSITIVITY_ONLY' if decision['supported_for_separate_followup'] else 'FAIL_FIXED_SHARED_DIRECTION_CLOSED',folds=12,source_decisions=15120,actual_NN_packets=ref._view_eval_packets,input_backward_packets=backward_packets,input_backward_calls=backward_calls,optimizer_updates=0,target_access=False,automatic_training=False))
  print('COMPLETE '+str(decision),flush=True)
 except Exception as error:d.write(out/'failure.json',dict(status='FAILED',error=repr(error),no_automatic_retry=True));raise
