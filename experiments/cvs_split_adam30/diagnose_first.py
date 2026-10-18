"""Bounded C01 first-clip diagnosis against the immutable R01 release.

This file changes no production implementation or tolerance. It runs the same
first source batch, adds two retained-graph total-loss VJPs, saves only gradient
comparison aggregates, and stops before core.after_clip or optimizer.step.
"""
from contextlib import contextmanager
import importlib.util
import inspect
import json
import math
from pathlib import Path
import sys
import time
import traceback
from types import MethodType

TRAINING_COMMIT='37cce29bb9bf65e372b681242705a1837728b3f2'
DIAGNOSTIC_RUN='20261019-diagnostic-FcrossRX-splitAdam-firstclip-s2026092701-m1-r01'
RTOL=5e-4
ATOL=5e-7


class IntentionalDiagnosticComplete(Exception):
 pass


def bootstrap():
 here=Path(__file__).resolve().parents[2]
 runtime=json.loads((here/'runtime.json').read_text(encoding='utf-8-sig'))
 if runtime['training_commit']!=TRAINING_COMMIT or runtime['run_id']!=DIAGNOSTIC_RUN:
  raise ValueError('Fixed diagnostic run or training identity differs')
 release=Path(runtime['training_release'])
 if (release/'release_commit.txt').read_text(encoding='utf8').strip()!=TRAINING_COMMIT:
  raise ValueError('Only the immutable published R01 training commit is allowed')
 spec=importlib.util.spec_from_file_location('SPLITADAM_R01_DIAGNOSTIC_BOOTSTRAP',release/'experiments/cvs_split_adam30/runner.py')
 module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);module.bootstrap()
 from experiments.cvs_split_adam30 import design as d
 if Path(d.ROOT)!=release or Path(d.PROJECT)!=Path(runtime['project']):
  raise ValueError('Bootstrapped training release or project differs')
 d.RUN=DIAGNOSTIC_RUN;d.BASE=Path(runtime['project'])/'runs'/DIAGNOSTIC_RUN
 if d.BASE.exists():raise FileExistsError('Diagnostic output root already exists')
 d.BASE.mkdir(parents=True,exist_ok=False)
 return d


def _native_state():
 """Read a bounded scalar witness from the still-active native train frame."""
 frame=inspect.currentframe();result={};native=sys.modules.get('SSDG.train_ssdg');train_globals=getattr(getattr(native,'train',None),'__globals__',None)
 try:
  while frame is not None:
   if (frame.f_globals is train_globals or frame.f_globals.get('__name__')=='SSDG.train_ssdg') and 'r3_optimizer_steps' in frame.f_locals:
    local=frame.f_locals
    for key in ('r3_optimizer_steps','epoch','batch_idx','phase'):
     value=local.get(key)
     if value is None or isinstance(value,(bool,int,float,str)):result[key]=value
    scaler=local.get('scaler')
    result['scaler_enabled']=bool(scaler.is_enabled()) if scaler is not None else None
    result['native_train_frame_found']=True
    return result
   frame=frame.f_back
  return dict(native_train_frame_found=False)
 finally:del frame


def _runtime_flags(d,out):
 args=d.read(out/'resolved_native_args.json')
 keys=set(('amp','max_grad_norm','grad_clip','clip_grad_norm','a1_runtime_fast','a1_runtime_disable_diagnostics'))
 keys.update(k for k in args if k.startswith('os_'))
 result={}
 for key in sorted(keys):
  if key not in args:continue
  value=args[key]
  if value is None or isinstance(value,(bool,int,float)) or isinstance(value,str) and len(value)<80 and not any(c in value for c in ('/','\\')):result[key]=value
 return result


def make_observer(d):
 import torch
 from experiments.cvs_split_adam30 import observer as production,core
 from experiments.cvs_rdrop_sinc_units.optimizer import QNAMES

 def tensor_copy(value):return None if value is None else value.detach().clone()

 def stats(actual,reference,parameter,scale=1.):
  original_a=torch.zeros_like(parameter) if actual is None else actual
  original_b=torch.zeros_like(parameter) if reference is None else reference
  native_a=original_a if scale==1. else original_a/scale;native_b=original_b if scale==1. else original_b/scale
  aa=original_a.detach().reshape(-1).double()/scale;bb=original_b.detach().reshape(-1).double()/scale
  finite=bool(torch.isfinite(aa).all() and torch.isfinite(bb).all())
  if not finite:return dict(finite=False,elements=parameter.numel(),actual_active=actual is not None,reference_active=reference is not None)
  error=aa-bb;aa2=float(torch.dot(aa,aa));bb2=float(torch.dot(bb,bb));ee2=float(torch.dot(error,error));dot=float(torch.dot(aa,bb));count=parameter.numel()
  return dict(finite=True,elements=count,actual_active=actual is not None,reference_active=reference is not None,max_abs=float(error.abs().max()) if count else 0.,relative_L2=math.sqrt(ee2/bb2) if bb2 else 0. if not ee2 else None,error_RMS=math.sqrt(ee2/count) if count else 0.,actual_RMS=math.sqrt(aa2/count) if count else 0.,reference_RMS=math.sqrt(bb2/count) if count else 0.,cosine=dot/math.sqrt(aa2*bb2) if aa2 and bb2 else None,failed_elements=int((~torch.isclose(native_a,native_b,rtol=RTOL,atol=ATOL)).sum()),failed_elements_production_equivalent_tolerance=int((~torch.isclose(original_a,original_b,rtol=RTOL,atol=ATOL)).sum()),sum_error_squared=ee2,sum_actual_squared=aa2,sum_reference_squared=bb2,sum_dot=dot)

 def combined(a,b,p):
  if a is None and b is None:return None
  result=torch.zeros_like(p)
  if a is not None:result.add_(a)
  if b is not None:result.add_(b)
  return result

 def aggregate(records,coordinate):
  groups={}
  for record in records:
   for comparison,value in record['comparisons'].items():groups.setdefault(comparison,[]).append(value[coordinate])
  result={}
  for key,values in groups.items():
   if not all(v['finite'] for v in values):result[key]=dict(finite=False);continue
   count=sum(v['elements'] for v in values);ee=sum(v['sum_error_squared'] for v in values);aa=sum(v['sum_actual_squared'] for v in values);bb=sum(v['sum_reference_squared'] for v in values);dot=sum(v['sum_dot'] for v in values)
   result[key]=dict(finite=True,elements=count,max_abs=max((v['max_abs'] for v in values),default=0.),relative_L2=math.sqrt(ee/bb) if bb else 0. if not ee else None,error_RMS=math.sqrt(ee/count) if count else 0.,actual_RMS=math.sqrt(aa/count) if count else 0.,reference_RMS=math.sqrt(bb/count) if count else 0.,cosine=dot/math.sqrt(aa*bb) if aa and bb else None,failed_elements=sum(v['failed_elements'] for v in values),failed_elements_production_equivalent_tolerance=sum(v['failed_elements_production_equivalent_tolerance'] for v in values),parameters_with_failures=sum(v['failed_elements']>0 for v in values))
  return result

 class CountRoleLoader:
  def __init__(self,loader,role,counts):self.loader=loader;self.role=role;self.counts=counts
  def __getattr__(self,key):return getattr(self.loader,key)
  def __len__(self):return len(self.loader)
  def __iter__(self):
   for batch in self.loader:
    x=batch[0]
    self.counts[self.role]['batches']+=1;self.counts[self.role]['packets']+=len(x)
    yield batch

 class DiagnosticObserver(production.Observer):
  def __init__(self,c):
   self.diagnostic_stage='construct_original_observer';self.started=time.perf_counter();self.loader_counts={r:dict(batches=0,packets=0) for r in ('L','U')};self.extra_diagnostic_VJP_calls=0
   super().__init__(c)
  def data(self,native,ctx,out):
   self.diagnostic_stage='source_data';super().data(native,ctx,out)
   for key,role in [('train_loader','L'),('unlabeled_loader','U')]:ctx[key]=CountRoleLoader(ctx[key],role,self.loader_counts)
  def initialize(self,m,c,out):
   self.diagnostic_stage='original_initialize';super().initialize(m,c,out);original=m._source_optimizer_update_probe_hook
   def repeated_probe(this,**kw):
    self.diagnostic_stage='original_core_capture';result=original(**kw)
    pending=this._sa_pending
    if pending is None or pending[4]!=1 or kw['optimizer_step']!=1:raise ValueError('Only the first captured native update is permitted')
    named=pending[0];params=tuple(p for _,p in named)
    self.diagnostic_stage='repeat_total_loss_VJP_1';first=torch.autograd.grad(kw['total_loss'],params,retain_graph=True,allow_unused=True)
    self.diagnostic_stage='repeat_total_loss_VJP_2';second=torch.autograd.grad(kw['total_loss'],params,retain_graph=True,allow_unused=True)
    self.extra_diagnostic_VJP_calls=2;self.direct_VJP1=tuple(tensor_copy(v) for v in first);self.direct_VJP2=tuple(tensor_copy(v) for v in second)
    if any(p.grad is not None for p in params):raise ValueError('Repeated VJPs altered native gradient accumulation')
    self.probe_summary=dict(optimizer_step=int(kw['optimizer_step']),epoch=int(kw['epoch']),batch_idx=int(kw['batch_idx']),phase=str(kw['phase']),labeled_count=int(kw['labeled_count']),unlabeled_count=int(kw['unlabeled_count']),total_loss=float(kw['total_loss'].detach()),unlabeled_loss=float(kw['unlabeled_loss'].detach()))
    self.diagnostic_stage='await_real_backward';return result
   m._source_optimizer_update_probe_hook=MethodType(repeated_probe,m)
  @contextmanager
  def native_scope(self,native):
   original_after_clip=core.after_clip
   with super().native_scope(native):
    original_clip=torch.nn.utils.clip_grad_norm_
    def before_clip(parameters,max_norm,*args,**kw):
     self.diagnostic_stage='before_original_physical_clip';params=list(parameters)
     named=self.model._sa_pending[0]
     if len(params)!=len(named) or {id(p) for p in params}!={id(p) for _,p in named}:raise ValueError('Original complete parameter clip set differs')
     self.backward_preclip=tuple(tensor_copy(p.grad) for _,p in named)
     self.diagnostic_stage='original_physical_clip';return original_clip(params,max_norm,*args,**kw)
    torch.nn.utils.clip_grad_norm_=before_clip;core.after_clip=self.diagnostic_after_clip
    try:yield
    finally:torch.nn.utils.clip_grad_norm_=original_clip;core.after_clip=original_after_clip
  def diagnostic_after_clip(self,m,parameters,max_norm,total_norm):
   self.diagnostic_stage='aggregate_before_original_core_assert';pending=m._sa_pending
   if m is not self.model or pending is None or float(max_norm)!=5. or pending[4]!=1 or self.c['arm']!='C01' or m._sa_mode!='MEASURE':raise ValueError('Only C01 original first physical clip is permitted')
   named,rest,satellite,geometry,step,capture_seconds=pending;opt=self.optimizer
   norm=float(total_norm);factor=min(1.,5./(norm+1e-6));records=[]
   if not math.isfinite(norm):raise FloatingPointError('Nonfinite original physical clip norm')
   with torch.no_grad():
    for index,((name,p),r,a,one,two,before) in enumerate(zip(named,rest,satellite,self.direct_VJP1,self.direct_VJP2,self.backward_preclip)):
     split_sum=combined(r,a,p);clipped_r=None if r is None else r.mul(factor);clipped_a=None if a is None else a.mul(factor);production_expected=combined(clipped_r,clipped_a,p);before_scaled=None if before is None else before.mul(factor)
     pairs=dict(repeated_direct_VJP=(two,one),split_sum_vs_direct=(split_sum,one),real_backward_preclip_vs_direct=(before,one),real_backward_postclip_vs_preclip_scaled=(p.grad,before_scaled),production_split_clipped_vs_real_backward=(production_expected,p.grad))
     scale=25000. if name in QNAMES else 1.
     records.append(dict(parameter=name,shape=list(p.shape),elements=p.numel(),Sinc_physical_scale=scale,rest_active=r is not None,As_active=a is not None,direct1_active=one is not None,direct2_active=two is not None,native_preclip_active=before is not None,native_postclip_active=p.grad is not None,comparisons={key:dict(storage=stats(actual,reference,p),physical=stats(actual,reference,p,scale)) for key,(actual,reference) in pairs.items()}))
   native=_native_state();parameter_equal=all(torch.equal(p.detach().cpu(),self.parent['model'][name]) for name,p in m.named_parameters());teacher_equal=all(torch.equal(p.detach().cpu(),self.parent['ema_model'][name]) for name,p in self.teacher.named_parameters())
   if opt._sinc_unit_steps!=0 or m._sa_steps!=0 or m._sa_clips!=0 or m._sa_vjp_calls!=2 or opt.state or opt._sinc_physical_clip_calls!=1 or not parameter_equal or not teacher_equal:raise ValueError('Diagnosis reached or mutated a successful optimizer/EMA update')
   if not native.get('native_train_frame_found') or native.get('r3_optimizer_steps')!=0 or native.get('scaler_enabled') is not False:raise ValueError('Native successful-update/AMP witness unavailable or differs')
   e=m.id_backbone.encoder;t=self.teacher.id_backbone.encoder
   counters=dict(student_train_NN=int(e._full_train_packets),student_eval_NN=int(e._full_eval_packets),teacher_train_NN=int(t._full_train_packets-self.teacher_offset['train']),teacher_eval_NN=int(t._full_eval_packets-self.teacher_offset['eval']),teacher_counter_offset=dict(self.teacher_offset),L_NN=int(m._full_L_packets+m._shape_NN),L_As=int(m._full_channel.packets),U_probe_physical_samples=int(self.probe_summary['unlabeled_count']),loader_counts=self.loader_counts,core_VJP_calls=int(m._sa_vjp_calls),additional_diagnostic_VJP_calls=self.extra_diagnostic_VJP_calls,total_VJP_calls=int(m._sa_vjp_calls)+self.extra_diagnostic_VJP_calls,successful_optimizer_updates=0,successful_EMA_updates=0,Sinc_physical_clip_calls=int(opt._sinc_physical_clip_calls),Sinc_unit_steps=int(opt._sinc_unit_steps))
   counters['total_NN']=sum(counters[k] for k in ('student_train_NN','student_eval_NN','teacher_train_NN','teacher_eval_NN'))
   if counters['L_NN']!=768 or counters['L_As']!=64 or counters['loader_counts']['L']!=dict(batches=1,packets=128) or counters['loader_counts']['U']['batches']!=1 or counters['U_probe_physical_samples']!=counters['loader_counts']['U']['packets']:raise ValueError('Expected original first-batch L/As/U physical cost differs')
   value=dict(status='DIAGNOSTIC_COMPLETE_ZERO_UPDATES',run_id=DIAGNOSTIC_RUN,training_commit=TRAINING_COMMIT,row_id=self.c['row_id'],optimizer_mode='MEASURE',scope='one original C01 source batch;stops before production core.after_clip and any optimizer/EMA step',stage=self.diagnostic_stage,production_tolerance=dict(rtol=RTOL,atol=ATOL,storage_coordinates=True),physical_metric_note='QNAMES storage gradients divided by25000;failed_elements uses same numeric tolerance,production_equivalent count uses original storage dtype and tolerance',physical_global_norm=norm,clip_factor=factor,native_flags=_runtime_flags(d,self.out),native_frame=native,probe=self.probe_summary,core_capture_geometry=geometry,core_capture_host_seconds=capture_seconds,parameter_count=len(records),parameters_equal_parent=True,teacher_parameters_equal_parent_EMA=True,cost=counters,aggregate_storage=aggregate(records,'storage'),aggregate_physical=aggregate(records,'physical'),parameters=records,elapsed_native_wall_seconds=time.perf_counter()-self.started,real_source_IQ=True,source_RX_holdout=None,U_TX_truth_read=False,target_query_read=False,target_truth_read=False,gradient_arrays_saved=False,IQ_saved_in_diagnostic=False,physical_IDs_saved_in_diagnostic=False,interpretation='Diagnostic measurements only;no numerical cause presumed and no tolerance or production code changed;original private source artifacts retained')
   d.write(d.BASE/'diagnostic.json',value);self.diagnostic_stage='intentional_stop_before_optimizer';raise IntentionalDiagnosticComplete()
 return DiagnosticObserver


def main():
 d=None;observer=None;started=time.perf_counter()
 try:
  d=bootstrap()
  from experiments.cvs_split_adam30 import model
  from experiments.cvs_split_adam30.source import segment_train
  row=next(r for r in d.rows() if r['arm']=='C01');config=d.config(row);observer=make_observer(d)(config)
  segment_train(config,design=d,model_api=model,observer=observer)
  raise RuntimeError('Diagnostic unexpectedly returned from original full training')
 except IntentionalDiagnosticComplete:
  if d is None or not (d.BASE/'diagnostic.json').is_file():raise RuntimeError('Intentional stop has no diagnostic artifact')
  d.write(d.BASE/'complete.json',dict(status='DONE',run_id=DIAGNOSTIC_RUN,training_commit=TRAINING_COMMIT,diagnostic='diagnostic.json',successful_optimizer_updates=0,successful_EMA_updates=0,elapsed_wall_seconds=time.perf_counter()-started,target_access=False))
  print('FIRST_C01_CLIP_DIAGNOSTIC DONE;optimizer_updates=0;EMA_updates=0',flush=True);return 0
 except Exception as exc:
  if d is not None:
   d.write(d.BASE/'failure.json',dict(status='FAILED',stage=getattr(observer,'diagnostic_stage','bootstrap_or_observer_construction'),exception_type=type(exc).__name__,message=str(exc),elapsed_wall_seconds=time.perf_counter()-started,diagnostic_complete=False))
   (d.BASE/'failure.traceback.txt').write_text(traceback.format_exc(),encoding='utf8')
  raise


if __name__=='__main__':sys.exit(main())
