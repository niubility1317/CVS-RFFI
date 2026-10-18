"""Focused pure-CPU protocol and real-native synthetic augmentation checks; no NN/CP/IQ."""
import ast,copy,inspect,sys,types,random,json
from pathlib import Path
import numpy as np
import torch
from . import design as d
from . import augmentation as a
from .protocol import cumulative,check_checkpoint,check_default_head

def run():
 negative=0
 def bad(fn):
  nonlocal negative
  try:fn()
  except (ValueError,TypeError,KeyError,StopIteration):negative+=1
  else:raise AssertionError('Invalid P78 path accepted')
 for row in d.rows():
  c=d.config(row);d.validate(c);assert c['As']==(row['arm']=='B01') and Path(c['parent_source'])==Path(d.reference_config(row,30)['parent_source']) and [Path(z) for z in c['checkpoint_sources']]==[Path(c['parent_source'])/'final_ssdg.pth']
  assert c['epochs']==10 and c['updates']==2220 and c['cumulative_updates']==6660 and c['training_mechanism']['pair_weights']==[.5,.5,.25] and c['crossview']['mode']=='ORD128' and c['ordinary_channel']['removed']==['AWGN','MP']
  for k,v in [('target_access',True),('As',not c['As']),('epochs',30),('updates',6660),('parent_source','SAT18'),('step_offset',0),('separate_head_training',True),('model_seed',2026092702)]:
   x=copy.deepcopy(c);x[k]=v;bad(lambda x=x:d.validate(x))
 assert cumulative(1,1)==(21,4441) and cumulative(10,2220)==(30,6660)
 bad(lambda:cumulative(1,223));bad(lambda:cumulative(10,2221));bad(lambda:d.config(d.rows()[0],20));check_default_head((6,160));bad(lambda:check_default_head((6,64)))
 import DataAugmentation as native
 torch.set_num_threads(2)
 aug=native.RFFIAugmentor(p_dac=0.,p_pa=.15,enable_class_signature=False,p_multipath=1.,p_awgn=1.,mp_taps_min=2,mp_taps_max=4,mp_delay_max=1,max_time_shift=3,p_time_shift=.2,p_phase_rot=.3,p_cfo=.2,p_phase_noise=.2,p_amp_scale=.3,p_dc_offset=.1,p_bandedge_taper=.1)
 g=torch.Generator().manual_seed(81078);x=torch.randn((128,2,256),generator=g);y=torch.arange(128)%6;saved=x.clone();device=x.device
 random.seed(81178);np.random.seed(81278);torch.manual_seed(81378);start=a.rng_state(device);MP,AWGN=native._apply_multipath,native._apply_awgn
 full=aug(x.clone(),labels=y,no_pa=False);end=a.rng_state(device)
 a.restore_rng(start,device);off,offrec=a.draw(aug,x,y,enabled=False)
 assert torch.equal(full,off) and a.same_rng(end,a.rng_state(device)) and offrec['MP_calls']==offrec['AWGN_calls']==1
 # Independent direct native primitive replay/reference, with all other operations kept.
 a.restore_rng(start,device)
 def mp(z,*args,**kw):MP(z,*args,**kw);return z
 def awgn(z,*args,**kw):AWGN(z,*args,**kw);return z
 try:
  native._apply_multipath=mp;native._apply_awgn=awgn;expected=aug(x.clone(),labels=y,no_pa=False)
 finally:native._apply_multipath=MP;native._apply_awgn=AWGN
 expected_end=a.rng_state(device);assert a.same_rng(end,expected_end)
 a.restore_rng(start,device);actual,rec=a.draw(aug,x,y)
 assert torch.equal(actual,expected) and torch.equal(x,saved) and a.same_rng(end,a.rng_state(device)) and rec['MP_calls']==rec['AWGN_calls']==1 and rec['shadow_MP_packets']==rec['shadow_AWGN_packets']==128 and rec['primitive_restored'] and rec['post_return_rng_preserved']
 # OFF exact path + same-RNG L execution leaves native U256 and last124 outputs exact.
 native_tree=ast.parse((Path(native.__file__).parent/'SSDG/train_ssdg.py').read_text(encoding='utf8'))
 strong_node=next(n for n in native_tree.body if isinstance(n,ast.FunctionDef) and n.name=='_strong_augment');env={'torch':torch};exec(compile(ast.Module(body=[strong_node],type_ignores=[]),'actual_native_U_strong','exec'),env);strong=env['_strong_augment']
 calls=[n for n in ast.walk(native_tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='augmentor']
 assert len(calls)==2 and all(isinstance(n.args[0],ast.Name) and n.args[0].id=='x_l' for n in calls) and not any(isinstance(n,ast.Name) and n.id=='augmentor' for n in ast.walk(strong_node))
 ur=[];guard=a.strong_passthrough(strong,ur,lambda:1)
 u=torch.randn((256,2,256),generator=g)
 a.restore_rng(end,device);u0=strong(u,.015);u1=strong(u[:124],.015);uend=a.rng_state(device)
 a.restore_rng(expected_end,device);v0=guard(u,.015);v1=guard(u[:124],.015)
 assert torch.equal(u0,v0) and torch.equal(u1,v1) and a.same_rng(uend,a.rng_state(device)) and [r['packets'] for r in ur]==[256,124] and all(r['native_output_passthrough'] and r['L_primitive_wrappers_inactive'] for r in ur)
 records=[];factory=a.ordinary_factory(lambda *args,**kw:aug,records);wrapped=factory({});a.restore_rng(start,device);out=wrapped(x,labels=y,no_pa=False)
 assert torch.equal(out,actual) and factory.calls==1 and len(records)==1 and records[0]['count']==128
 for call in [lambda:factory({}),lambda:wrapped(x,labels=None,no_pa=False),lambda:wrapped(x,labels=y,no_pa=True),lambda:wrapped(x[:127],labels=y[:127],no_pa=False),lambda:wrapped(x,labels=y+7,no_pa=False),lambda:guard(u[:128],.015)]:bad(call)
 # Failure restores both module functions and scope without changing another object.
 def fail(z,*args,**kw):raise ValueError('synthetic native primitive failure')
 native._apply_awgn=fail
 try:bad(lambda:a.draw(aug,x,y));assert native._apply_multipath is MP and native._apply_awgn is fail and not a.patch_active()
 finally:native._apply_awgn=AWGN
 assert native._apply_multipath is MP and native._apply_awgn is AWGN
 # With both primitive gates off, the entire ordinary result is native bitwise exact.
 aug.p_multipath=aug.p_awgn=0.;s=a.rng_state(device);native_out=aug(x.clone(),labels=y,no_pa=False);e=a.rng_state(device);a.restore_rng(s,device);same,r=a.draw(aug,x,y)
 assert torch.equal(same,native_out) and a.same_rng(e,a.rng_state(device)) and not r['MP_hit'] and not r['AWGN_hit']
 # Execute the actual telemetry hook around an unchanged differentiable original return.
 from .schema import CONCAT_FIELDS,OC_FIELDS,INFO_KEYS
 tree=ast.parse(Path(__file__).with_name('model.py').read_text(encoding='utf8'));node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='hook')
 loss=torch.tensor(2.,requires_grad=True);env={'original':types.SimpleNamespace(hook=lambda self,**kw:(loss,{})),'CONCAT_FIELDS':CONCAT_FIELDS,'OC_FIELDS':OC_FIELDS}
 exec(compile(ast.Module(body=[node],type_ignores=[]),'actual_P78_hook','exec'),env)
 ordinary=dict(rec,step=1);record=dict(total_L=1.25,common_CE=1.,common_KL=.1,SupCon=.2,SupCon_weight=.1,L_forwards=6,As_packets=64,crossview_mode='ORD128')
 model=types.SimpleNamespace(_full_records=[record],_oc_ordinary_records=[ordinary],_oc_records=[])
 delta,info=env['hook'](model,optimizer_step=1);assert delta is loss and set(info)==set(INFO_KEYS) and len(info)==18 and info['ordinary_channel/MP_packets']==128
 delta.backward();assert loss.grad.item()==1.
 from . import report
 assert report.PAIRS==(('OC_AS30','OC_NOAS30'),('OC_AS30','AS30'),('OC_NOAS30','NOAS30'),('AS30','NOAS30'))
 for file in Path(__file__).parent.glob('*.py'):ast.parse(file.read_text(encoding='utf8'),filename=str(file))
 json.dumps(rec,allow_nan=False)
 return dict(status='PASS',protocol_negatives=negative,actual_native_OFF_output_and_RNG_exact=True,actual_native_forced_MP_AWGN_bypass_exact=True,all_other_native_ops_PA_retained=True,U256_last124_output_and_RNG_exact=True,finally_restored=True,As_path_not_wrapped=True,original_P43_own20_6L_retained=True,epoch21_30_boundaries=True,NN=0,CP=0,real_IQ=0,updates=0,target_access=False)
if __name__=='__main__':print(json.dumps(run()))
