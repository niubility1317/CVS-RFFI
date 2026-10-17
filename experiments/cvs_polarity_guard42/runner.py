"""Frozen P41 ORD default head;no optimizer/teacher/U/target/As."""
import argparse,importlib.util,json,os,sys,time,traceback,types
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
def bootstrap():
 ref=Path('/home/szu2070436088/2510044040/CV-SincNet/releases/cvs_FcrossRX_crossviewSupCon_E30_20261018_r01')
 if (ref/'release_commit.txt').read_text().strip()!='ec4c5d7f46aedb5aaece77fdd72dbe6ed4c157f2':raise ValueError('Pinned real P41 control code differs')
 spec=importlib.util.spec_from_file_location('polarity42_control',ref/'experiments/cvs_crossview_supcon41/runner.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);m.bootstrap();p=types.ModuleType('experiments.cvs_polarity_guard42');p.__path__=[str(ROOT/'experiments/cvs_polarity_guard42')];sys.modules[p.__name__]=p

def run(device):
 from experiments.cvs_polarity_guard42 import design as d,ops,metrics
 if d.BASE.exists():raise FileExistsError('Preserve existing unique source diagnostic;no overwrite/relaunch')
 d.BASE.mkdir();tic=time.perf_counter();d.write(d.BASE/'resolved_config.json',d.config())
 try:
  import numpy as np,torch
  from experiments.cvs_crossview_supcon41 import design as b
  from experiments.cvs_crossview_supcon41.model import own_model
  from experiments.cvs_crossview_supcon41.quality import panel
  row=next(r for r in b.rows() if r['row_id']==d.REFERENCE_ROW);cfg=b.config(row);m=own_model(cfg,device)
  if any(p.requires_grad for p in m.parameters()) or any(z.training for z in m.modules()) or m.id_backbone.encoder._full_eval_packets!=0:raise ValueError('Frozen exact default source model required')
  head=m.id_backbone.encoder.core.id_backbone.cls_head
  if tuple(head.weight.shape)!=(6,160):raise ValueError('Existing joint160/default6 cosine head required')
  before={k:v.detach().cpu().clone() for k,v in m.state_dict().items()};g=torch.Generator().manual_seed(d.SMOKE_SEED);sx=torch.randn((2,2,256),generator=g)
  si=ops.invariant_summary(sx,ops.conjugate_iq(sx))
  with torch.no_grad():sm=m.forward_identity_only(sx.to(device))['tx_logits']
  if sm.shape!=(2,6) or not bool(torch.isfinite(sm).all()) or m.id_backbone.encoder._full_eval_packets!=2:raise ValueError('Real checkpoint synthetic no-query smoke failed')
  proof=dict(status='PASS',NN=2,source_IQ_packets=0,target_access=False,updates=0,head_fit_updates=0,default_head_class=type(head).__name__,default_head_shape=[6,160],synthetic_invariants=si)
  d.write(d.BASE/'real_checkpoint_smoke.json',proof);print('REAL_CHECKPOINT_NO_QUERY PASS2NN;continue source diagnostic',flush=True)
  data,roles=panel(torch);x=data['x'];reflected=ops.conjugate_iq(x);invariants=ops.invariant_summary(x,reflected);raw=np.empty((d.N,6),np.float32);conj=np.empty_like(raw)
  with torch.no_grad():
   for view,storage in [(x,raw),(reflected,conj)]:
    for start in range(0,d.N,90):storage[start:start+90]=m.forward_identity_only(view[start:start+90].to(device))['tx_logits'].tolist()
  if any(not torch.equal(v,m.state_dict()[k].detach().cpu()) for k,v in before.items()) or any(p.grad is not None for p in m.parameters()) or m.id_backbone.encoder._full_eval_packets!=3602:raise ValueError('All state must remain frozen;exact sourceNN budget required')
  y=np.asarray(data['y'].tolist(),np.int64);rx=np.asarray(data['rx'].tolist(),np.int64);day=np.asarray(data['day'].tolist(),np.int64);decision=metrics.decide(y,raw,conj,rx);rows=metrics.stratified(y,raw,conj,rx,day)
  np.savez_compressed(d.BASE/'predictions.npz',ids=np.asarray(data['ids']),TX=y,RX=rx,day=day,raw=raw,conjugate=conj)
  d.write(d.BASE/'source_metrics.json',dict(rows=rows,macro_F1_convention='All6registeredclasses,zero_division0'))
  d.write(d.BASE/'source_decision.json',decision);d.write(d.BASE/'input_invariants.json',invariants)
  d.write(d.BASE/'checkpoint_provenance.json',dict(verdict='VERIFIED_SOURCE_CONTRACT',checkpoint=str(Path(cfg['output_root'])/'final_ssdg.pth'),selection='P41 preregistered ORD control/fixedfinal;not formal-ranked',own_stage20_and_EMA_strict=True,source_role_handles='EXACT_MATCH',parent_epochs=30,parent_updates=6660,new_epochs=0,new_updates=0,used_as_training_initializer=False,default_head_jointly_trained_in_parent=True,target_training_or_selection=False,target_exposure_context='Parent formal reporting exists;this prefixed control was fixed before those scores'))
  d.write(d.BASE/'data_roles.json',dict(role='V',packets=d.N,source_RX=list(d.RX),source_days=[1,2,3],source_TX=list(range(6)),balanced_per_TX_RX_day=20,source_role_handles='Original exactV1800 subset of V27000 verified by reused panel loader',L_IQ_consumed=0,U_TX_truth_read=False,V_updates=0,target_access=False))
  result=dict(status='SOURCE_DIAGNOSTIC_COMPLETE',source_decision=decision,NN=3602,source_NN=3600,synthetic_NN=2,optimizer_updates=0,head_fit_updates=0,teacher_NN=0,new_training_epochs=0,As_calls=0,model_seed_count=1,source_only=True,U_TX_truth_read=False,target_access=False,model_state_unchanged=True,actual_checkpoint_model_reads=1,source_cache_reads=1,source_waveforms_consumed=1800,wall_seconds=time.perf_counter()-tic,code_commit=(ROOT/'release_commit.txt').read_text().strip(),automatic_training=False,goal_complete=False)
  d.write(d.BASE/'complete.json',result);print(json.dumps(result),flush=True)
 except BaseException as e:
  d.write(d.BASE/'failure.json',dict(error=repr(e),traceback=traceback.format_exc(),preserve_outputs=True,no_auto_retry=True));raise

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--device',choices=['cpu','cuda:0'],default='cuda:0');a=p.parse_args();bootstrap();run(a.device)
