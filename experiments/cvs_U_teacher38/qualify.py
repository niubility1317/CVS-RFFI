"""Source-only qualification of both fixed endpoints; all weights remain frozen."""
import os,sys,time,traceback
from pathlib import Path
import numpy as np
import torch
from . import design as d
from .maths import teacher_summary

def metadata():
 expected=d.read(d.SOURCE);proof=[]
 for row in d.rows():
  b=row['source'];init=d.read(b/'initialization.json');args=d.read(b/'actual_native_args.json')
  terminal=d.read(b/('completion.json' if row['row_id']=='E20' else 'phase1_terminal_status.json'))
  d.validate_metadata(row,d.read(b/'source_contract.json'),expected,init,args,terminal)
  if row['row_id']=='E30':
   ema=d.read(b/'EMA_initialization.json')
   if not ema['strict_own_parent'] or ema['parent']!=str(d.E20) or ema['extra_teacher']:raise ValueError('EMA parent differs')
  proof.append(dict(row_id=row['row_id'],checkpoint=str(b/'final_ssdg.pth'),source_roles='EXACT_MATCH',selection='fixed final20 / own final10 cumulative30;no new selection',teacher='own EMA only',training_dependency='scratch E20 then own E10',old_technical_closure_failure_retained=row['row_id']=='E30',diagnostic_only=True,future_training_initializer=False))
 return proof

def load_models(row):
 from experiments.cvs_cleanretain37 import design as f,model as api
 cp=torch.load(row['source']/'final_ssdg.pth',map_location='cpu',weights_only=False);a=cp['args']
 expected_run=row['source'].parents[1].name
 if cp.get('run_id')!=expected_run or cp.get('candidate_id')!='B01-s2026092701' or cp.get('epoch')!=row['segment_epochs'] or cp.get('checkpoint_selection')!='final_only' or cp.get('a1_ema_successful_updates')!=row['segment_updates'] or not isinstance(cp.get('ema_model'),dict):raise ValueError('Fixed own checkpoint identity/EMA differs')
 if a.get('teacher_ckpt') or a.get('freeze_backbone') or a.get('source_fold_held_rx') is not None or (row['row_id']=='E20' and (not a.get('from_scratch') or a.get('baseline_ckpt'))) or (row['row_id']=='E30' and (a.get('from_scratch') or a.get('baseline_ckpt')!=str(d.E20/'final_ssdg.pth'))):raise ValueError('Checkpoint inheritance differs')
 models=[]
 for key in ('model','ema_model'):
  m=api.build(f.config(f.rows()[0]),'cpu');m.load_state_dict(cp[key],strict=True);m.eval().requires_grad_(False)
  h=m.id_backbone.encoder.core.id_backbone.cls_head
  if sum(p.numel() for p in m.parameters())!=900722 or len(m.state_dict())!=206 or h.weight.shape!=(6,160) or float(h.scale)!=30.:raise ValueError('Fixed F/default head differs')
  with torch.no_grad():z=m.forward_identity_only(torch.zeros(2,2,256))['tx_logits']
  if z.shape!=(2,6) or not bool(torch.isfinite(z).all()):raise ValueError('Real CP no-query smoke failed')
  models.append(m)
 return models

def run():
 if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('CPU-only source diagnostic')
 if d.BASE.exists():raise FileExistsError('Reconcile immutable existing run;never overwrite')
 d.BASE.mkdir();started=time.perf_counter();torch.set_num_threads(2)
 d.write(d.BASE/'launch.json',dict(pid=os.getpid(),cwd=os.getcwd(),argv=sys.argv,owner=d.OWNER,CPU_only=True,commit=(d.ROOT/'release_commit.txt').read_text().strip(),target_access=False))
 try:
  proof=metadata();d.write(d.BASE/'provenance.json',proof)
  from experiments.cvs_equivariant_identity.precision import numerical_context
  from experiments.cvs_fullsource_cycleorder30.probe import panel
  from experiments.cvs_fullsource_concat30_25.channel import tensor_iq
  with numerical_context(d.POLICY):
   panel_data=None;results=[]
   for row in d.rows():
    out=d.BASE/row['row_id'];out.mkdir();tic=time.perf_counter();student,teacher=load_models(row)
    d.write(out/'real_checkpoint_smoke.json',dict(status='PASS',strict_load=True,NN=4,source_IQ_before_first_smoke=0 if panel_data is None else 1800,target_query=False,training_updates=0,head_fit_updates=0))
    if panel_data is None:
     panel_data,roles=panel(torch);d.write(d.BASE/'source_panel.json',roles)
     obs=d.read(d.PANEL/'observations/completion.json')
     if obs['status']!='SOURCE_OBSERVATIONS_COMPLETE' or obs['target_access'] or obs['draws']!=2 or obs['scenes']!=list(d.SCENES) or obs['conditions'][0]!='full':raise ValueError('Original full source observations required')
     array=np.load(d.PANEL/'observations/iq.npy',mmap_mode='r',allow_pickle=False)
     if array.shape!=(2,3,11,1800,2,256):raise ValueError('Original source panel shape differs')
    data=panel_data;x=data['x'];y=np.asarray(data['y'].tolist(),np.int64);rx=np.asarray(data['rx'].tolist(),np.int64)
    before=[{k:v.clone() for k,v in m.state_dict().items()} for m in (student,teacher)]
    clean=np.empty((1800,6),np.float32);teach=np.empty_like(clean);sat=np.empty((2,3,1800,6),np.float32)
    with torch.no_grad():
     for i in range(0,1800,90):
      clean[i:i+90]=student.forward_identity_only(x[i:i+90])['tx_logits'].tolist();teach[i:i+90]=teacher.forward_identity_only(x[i:i+90])['tx_logits'].tolist()
     for draw in range(2):
      for si in range(3):
       for i in range(0,1800,90):sat[draw,si,i:i+90]=student.forward_identity_only(tensor_iq(array[draw,si,0,i:i+90],'cpu'))['tx_logits'].tolist()
    for m,state,count in zip((student,teacher),before,(12602,1802)):
     if m.id_backbone.encoder._full_eval_packets!=count or m.id_backbone.encoder._full_train_packets or any(p.grad is not None for p in m.parameters()) or any(not torch.equal(v,m.state_dict()[k]) for k,v in state.items()):raise ValueError('Frozen model state/NN changed')
    result,q=teacher_summary(teach,clean,sat,y,rx,d.RULE)
    np.savez_compressed(out/'predictions.npz',ids=np.asarray(data['ids']),TX=y,RX=rx,student_clean=clean,teacher_clean=teach,student_satellite=sat,q=q)
    result.update(row_id=row['row_id'],status='SOURCE_ANALYZED',NN=14404,training_updates=0,head_fit_updates=0,target_access=False,U_truth_read=False,U_records_read=0,state_unchanged=True,wall_seconds=time.perf_counter()-tic)
    d.write(out/'complete.json',result);results.append(result);print(__import__('json').dumps(result),flush=True)
   final=dict(status='ANALYZED',rows=results,source_qualified=all(r['source_qualified'] for r in results),rule=d.RULE,actual_new_NN=28808,source_unique_V=1800,model_seed_count=1,checkpoint_endpoints=2,independent_model_seeds=1,training_updates=0,head_fit_updates=0,channel_draws=2,new_channel_calls=0,target_access=False,U_truth_read=False,U_records_read=0,formal_evaluation='NOT_APPLICABLE_SOURCE_ONLY_NO_TRAINING',future_training='unlaunched;own first-U teacher still requires fixed source check;no old teacher initializer',goal_complete=False,wall_seconds=time.perf_counter()-started)
   d.write(d.BASE/'complete.json',final)
 except BaseException as e:
  d.write(d.BASE/'failure.json',dict(status='FAILED_TECHNICAL',message=str(e),traceback=traceback.format_exc(),preserve_outputs=True,no_automatic_retry=True));raise
if __name__=='__main__':run()
