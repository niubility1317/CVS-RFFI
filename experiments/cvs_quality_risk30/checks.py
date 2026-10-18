"""Focused finite-bank risk/label/parent negatives;no realIQ/CP/network forward."""
import ast,copy,json,random
from pathlib import Path
import numpy as np
from . import design as d
from .protocol import cumulative,check_checkpoint,check_source_metadata,check_default_head
from .schema import INFO_KEYS,CONCAT_FIELDS,BANK_FIELDS

def legacy_smoothing_source_check():
 import importlib.util
 try:spec=importlib.util.find_spec('experiments.cvs_sigtime_rdrop_legacy.model')
 except (ModuleNotFoundError,ValueError):spec=None
 origin=getattr(spec,'origin',None);p=Path(origin) if origin and origin not in ('built-in','frozen') else d.ROOT/'experiments/cvs_sigtime_rdrop_legacy/model.py'
 if p.is_file():
  assert 'smoothing!=.01' in p.read_text(encoding='utf8');return dict(status='PASS',source=str(p),locator='actual import spec' if origin else 'local static source')
 return dict(status='NOT_PACKAGED_NONBLOCKING',source=None,interpretation='Actual globalLS.01 and originalterms checked by launcher smoke')

def numpy_loss(a,b,y):
 def soft(z):z=z-z.max(1,keepdims=True);lp=z-np.log(np.exp(z).sum(1,keepdims=True));return lp,np.exp(lp)
 lp,p=soft(a);lq,q=soft(b);t=np.full_like(a,.01/6);t[np.arange(len(y)),y]+=.99;ce=-.5*(t*(lp+lq)).sum(1);diff=lp-lq;kl=.5*((p-q)*diff).sum(1)
 ga=.5*(p-t)+.5*(p*(diff-(p*diff).sum(1,keepdims=True))+p-q);dq=-diff;gb=.5*(q-t)+.5*(q*(dq-(q*dq).sum(1,keepdims=True))+q-p)
 return ce+kl,ce,kl,ga,gb

def finite_risk_checks():
 rng=np.random.default_rng(107);groups=np.repeat(np.arange(4),[1,2,3,4]);n=len(groups);a=rng.normal(size=(n,6));b=rng.normal(size=(n,6));y=np.arange(n)%6;ell,ce,kl,ga,gb=numpy_loss(a,b,y);mass=np.bincount(groups,minlength=4)/n
 expected=np.asarray([sum((16/64)*(4*mass[g])*(groups==g)/sum(groups==g) for g in range(4))]).reshape(n)
 assert np.allclose(expected,np.full(n,1/n),atol=1e-16,rtol=0)
 assert abs(sum(mass[g]*ell[groups==g].mean() for g in range(4))-ell.mean())<1e-14
 assert np.allclose(ga*expected[:,None],ga/n,atol=1e-16,rtol=0) and np.allclose(gb*expected[:,None],gb/n,atol=1e-16,rtol=0)
 fd_error=0.
 for side in (0,1):
  for row,col in ((0,0),(2,4),(8,1)):
   aa=a.copy();bb=b.copy();arr=aa if side==0 else bb;eps=1e-6;arr[row,col]+=eps;hi=numpy_loss(aa,bb,y)[0].sum();arr[row,col]-=2*eps;lo=numpy_loss(aa,bb,y)[0].sum();truth=(ga if side==0 else gb)[row,col];fd_error=max(fd_error,abs((hi-lo)/(2*eps)-truth))
 assert fd_error<2e-8
 # Original mixed outer.5/128:ordinary64 andbank64 eachkeep absolute .25.
 assert .5/128*64==.25 and .5+.5+.25==1.25
 return dict(status='PASS',unequal_bins=[1,2,3,4],exact_expected_item_mass=True,expected_loss_and_gradient_match_finite_bank=True,max_gradient_finite_difference_error=float(fd_error),CE_and_KL_same_weights=True,no_self_normalization=True,As_outer_weight=.25,L_scale=1.25)

def sampling_checks():
 from .bank import select_indices,bins,stable_seed,NAMESPACE
 groups=np.repeat(np.arange(4),[1,2,3,4]);negative=0
 def bad(f):
  nonlocal negative
  try:f()
  except (ValueError,TypeError):negative+=1
  else:raise AssertionError('Illegal finite bank sampling accepted')
 before=np.random.get_state();py=random.getstate()
 for mode in ('IID','STRATIFIED'):
  index,w,r=select_indices(groups,mode,21,4441);other=select_indices(groups,mode,21,4441)
  assert np.array_equal(index,other[0]) and np.array_equal(w,other[1]) and len(index)==64 and np.all(index>=0) and np.all(index<len(groups)) and abs(w.sum()-64)<1e-12
  assert r['with_replacement'] and not r['global_RNG_used'] and len(set(index.tolist()))<64
  if mode=='IID':assert np.array_equal(w,np.ones(64))
  else:assert r['bin_counts']==[16]*4 and np.allclose(w,4*np.asarray(r['nominal_bin_mass'])[groups[index]],atol=0,rtol=0)
 after=np.random.get_state();assert before[0]==after[0] and np.array_equal(before[1],after[1]) and before[2:]==after[2:] and py==random.getstate()
 assert bins(np.asarray([-100.,2.999,3.,7.999,8.,14.999,15.,100.])).tolist()==[0,0,1,1,2,2,3,3]
 for f in (lambda:select_indices(np.zeros(10,int),'IID',21,4441),lambda:select_indices(groups,'OTHER',21,4441),lambda:select_indices(groups,'IID',1,1),lambda:select_indices(np.asarray([0,1,2,4]),'IID',21,4441),lambda:bins(np.asarray([np.nan]))):bad(f)
 return dict(status='PASS',negatives=negative,IID_and_stratified_private_reproducible=True,with_replacement_handles_singletons=True,global_NumPy_Python_RNG_unchanged=True,bin_boundaries_exact=True)

def actual_target_boundary():
 class Tensor(np.ndarray):
  def clone(self):return self.copy().view(Tensor)
 def t(x):return np.asarray(x).view(Tensor)
 tree=ast.parse(Path(__file__).with_name('model.py').read_text(encoding='utf8'));fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='base_hook');nodes=[]
 for n in fn.body:
  if isinstance(n,ast.Assign):
   target=n.targets[0]
   if isinstance(target,ast.Name) and target.id in ('mixed_y','mixed_domains') or isinstance(target,ast.Subscript) and isinstance(target.value,ast.Name) and target.value.id in ('mixed_y','mixed_domains'):nodes.append(n)
 assert len(nodes)==4
 f=ast.parse('def actual(labels,domains,smask,bank_y,bank_domains): pass').body[0];f.body=nodes+[ast.Return(value=ast.Tuple(elts=[ast.Name(id=k,ctx=ast.Load()) for k in ('mixed_y','mixed_domains')],ctx=ast.Load()))];env={};exec(compile(ast.fix_missing_locations(ast.Module(body=[f],type_ignores=[])),'actual_mixed_target_boundary','exec'),env)
 y=t(np.arange(128)%6);domains=t(np.arange(128)%15);mask=(np.arange(128)+1)%2!=0;by=t((y[mask]+1)%6);bd=t((domains[mask]+3)%15);yy,dd=env['actual'](y,domains,mask,by,bd)
 assert np.array_equal(yy[mask],by) and np.array_equal(dd[mask],bd) and np.array_equal(yy[~mask],y[~mask]) and np.array_equal(dd[~mask],domains[~mask]) and np.array_equal(y,np.arange(128)%6) and np.all(yy[mask]!=y[mask])
 text=Path(__file__).with_name('model.py').read_text(encoding='utf8');assert 'weighted_terms(z0,z1,mixed_y,weights)' in text and 'relations.penalty(a,b,labels,rx,smask,self._crossview_mode)' in text and 'view[smask]=satellite' in text and 'self._full_channel.apply' not in text
 return dict(status='PASS',actual_AST_mixed_labels_and_domains_replace_selected_only=True,bank_labels_differ_original=True,current_labels_not_modified=True,current_ORD128_metadata_unchanged=True,NN=0)

def torch_checks():
 try:import torch
 except ImportError:return dict(status='NOT_AVAILABLE_LOCAL',actual_weighted_autograd_and_Torch_RNG='remote focused checks andlauncher first770NN',NN=0)
 from .bank import select_indices
 from .losses import weighted_terms
 from experiments.cvs_sigtime_rdrop_legacy.model import terms
 state=torch.random.get_rng_state().clone();select_indices(np.repeat(np.arange(4),[1,2,3,4]),'STRATIFIED',21,4441);assert torch.equal(state,torch.random.get_rng_state())
 generator=torch.Generator().manual_seed(109);a=torch.randn(128,6,generator=generator,requires_grad=True);b=torch.randn(128,6,generator=generator,requires_grad=True);y=torch.arange(128)%6;one=torch.ones(128);new=weighted_terms(a,b,y,one);old=terms(a,b,y,1.)
 assert all(torch.allclose(x,z,atol=1e-6,rtol=1e-6) for x,z in zip(new[:3],old[:3]))
 gn=torch.autograd.grad(new[0],(a,b),retain_graph=True);go=torch.autograd.grad(old[0],(a,b));assert all(torch.allclose(x,z,atol=1e-7,rtol=1e-5) for x,z in zip(gn,go))
 return dict(status='PASS',actual_helper_allones_recovers_old_loss_gradient=True,Torch_sampling_RNG_unchanged=True,NN=0,synthetic_CE_operator_calls=4,synthetic_CE_logit_packets=512)

def interop_recovery_checks():
 """Direct r01 boundary/recovery regression only;no CP/IQ/NN/network access."""
 from types import SimpleNamespace
 from .recovery import validate_layout,HISTORY
 class Tensor(np.ndarray):
  def to(self,*args,**kw):return self
 calls=[]
 def frombuffer(data,dtype):calls.append(('bytes',len(data),dtype));return np.frombuffer(data,dtype=np.float32).view(Tensor)
 fake=SimpleNamespace(frombuffer=frombuffer,float32=np.float32)
 tree=ast.parse(Path(__file__).with_name('channel.py').read_text(encoding='utf8'));fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='tensor_iq');env=dict(np=np,torch=fake);exec(compile(ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[])),'actual_byte_IQ_bridge','exec'),env)
 original=np.asarray([0.,-0.,1.,-1.,np.finfo(np.float32).tiny,123.5],np.float32);array=np.tile(original,((64*2*256+5)//6))[:64*2*256].reshape(64,2,256);noncontiguous=array[:,:,::-1];out=env['tensor_iq'](noncontiguous,'cpu')
 assert out.shape==(64,2,256) and out.dtype==np.float32 and out.tobytes()==np.ascontiguousarray(noncontiguous).tobytes() and calls==[('bytes',64*2*256*4,np.float32)]
 for name in ('model.py','bank.py'):
  text=Path(__file__).with_name(name).read_text(encoding='utf8');assert 'from_numpy' not in text and '.numpy()' not in text
 modeltext=Path(__file__).with_name('model.py').read_text(encoding='utf8');banktext=Path(__file__).with_name('bank.py').read_text(encoding='utf8')
 assert "bmeta['TX'].tolist(),device=labels.device,dtype=torch.long" in modeltext and "bmeta['domain'].tolist(),device=domains.device,dtype=torch.long" in modeltext and "bmeta['RX'].tolist(),device=domains.device,dtype=torch.long" in modeltext and "ids[begin:end,j].tolist(),dtype=torch.long" in banktext and 'raw[i]=x.tolist()' in banktext and 'iq[sl]=out.tolist()' in banktext
 prior='PROJECT/releases/cvs_FcrossRX_qualityRisk_E30_20261018_r01';failure=dict(status='PREFLIGHT_FAILED',error="TypeError('expected np.ndarray (got numpy.ndarray)')",preserve_outputs=True,no_automatic_retry=True);dispatcher=dict(pid=1596070,cwd=prior,argv=[prior+'/experiments/cvs_quality_risk30/runner.py'],owner='codex/root/practical-qualityRisk');names=['dispatcher.json','failure.json','logs'];cost=validate_layout(names,[],failure,dispatcher,prior,False);assert cost['actual_NN']==258 and cost['optimizer_updates']==0 and cost['real_IQ']==0 and not cost['bank_generated'] and HISTORY=='technical_history/r01' and d.RELEASE.endswith('_r02')
 negative=0
 def bad(f):
  nonlocal negative
  try:f()
  except ValueError:negative+=1
  else:raise AssertionError('Unsafe recovery accepted')
 for extra in ('proposal_bank','real_checkpoint_smoke.json','I01-s2026092701','launch.json','technical_history'):bad(lambda extra=extra:validate_layout(names+[extra],[],failure,dispatcher,prior))
 bad(lambda:validate_layout(names,['worker.log'],failure,dispatcher,prior));bad(lambda:validate_layout(names,[],failure,dispatcher,prior,True))
 for key,value in [('status','TRAIN_FAILED'),('error','other'),('preserve_outputs',False)]:
  f=dict(failure);f[key]=value;bad(lambda f=f:validate_layout(names,[],f,dispatcher,prior))
 for key,value in [('pid',999),('cwd','otherrelease'),('argv',['other']),('owner','other')]:
  p=dict(dispatcher);p[key]=value;bad(lambda p=p:validate_layout(names,[],failure,p,prior))
 for f in Path(__file__).parent.glob('*.py'):ast.parse(f.read_text(encoding='utf8'),filename=str(f))
 return dict(status='PASS',targeted_negatives=negative,actual_IQ_byte_bridge_bitwise_float32=True,noncontiguous_input=True,metadata_explicit_long_lists=True,no_candidate_from_numpy_or_numpy_method=True,known_failure_NN=258,recovery_history=HISTORY,zero_updates_only=True,old_outputs_preserved=True,release=d.RELEASE,NN=0,CP=0,IQ=0,network_calls=0)

def run():
 negative=0
 def bad(f):
  nonlocal negative
  try:f()
  except (ValueError,TypeError,KeyError,StopIteration):negative+=1
  else:raise AssertionError('Invalid fixed scope accepted')
 assert [r['arm'] for r in d.rows()]==['I01','S01'];parents=[]
 for row in d.rows():
  c=d.config(row);d.validate(c);parents.append(c['parent_source']);assert c['reference_row_id']==d.REFERENCE_ROW and d.reference_config(row,20)['row_id']==d.REFERENCE_ROW and d.reference_config(row,30)['parent_source']==c['parent_source']
  assert c['data_contract']==dict(RX=[1,3,4,6,8],days=[1,2,3],L=6300,U=56700,V=27000,split=392005,source_RX_holdout=None,internal_L_split=False,U_TX_truth_hidden=True)
  assert (c['epochs'],c['updates'],c['cumulative_epochs'],c['cumulative_updates'])==(10,2220,30,6660) and c['training_mechanism']['pair_weights']==[.5,.5,.25] and c['crossview']['mode']=='ORD128'
  for key,value in [('reference_row_id','B00-s2026092701'),('parent_source','unknown'),('epochs',20),('step_offset',0),('As',False),('model_seed',2026092702),('label_epochs',13),('target_access',True)]:
   z=copy.deepcopy(c);z[key]=value;bad(lambda z=z:d.validate(z))
  for key,value in [('bank_size',6300),('CE_label_smoothing',.05),('KL_same_weights_as_CE',False),('mixed_labels_replaced',False),('global_RNG_used',True),('bank_weight','clipped')]:
   z=copy.deepcopy(c);z['quality_risk'][key]=value;bad(lambda z=z:d.validate(z))
 assert len(set(parents))==1;ctx=d.bank_contexts();assert len(ctx)==40 and len({(c['cumulative_epoch'],c['cumulative_step']) for c in ctx})==40 and all(c['cumulative_epoch']==(c['cumulative_step']-1)//222+1 for c in ctx)
 check_default_head((6,160));bad(lambda:check_default_head((6,64)));bad(lambda:d.config(d.rows()[0],20));assert cumulative(1,1)==(21,4441) and cumulative(10,2220)==(30,6660)
 cp=dict(epoch=20,a1_ema_successful_updates=4440,run_id=d.REFERENCE_RUN,candidate_id=d.REFERENCE_ROW,checkpoint_selection='final_only',ema_model={'mock':1},args=dict(seed=d.SEED,baseline_ckpt='',teacher_ckpt='',epochs=20,use_ema_teacher=True,label_smoothing=.01,batch_size=128));verify=lambda value:check_checkpoint(value,run=d.REFERENCE_RUN,row=d.REFERENCE_ROW,epoch=20,updates=4440,baseline='');verify(cp)
 for key,value in [('candidate_id',d.rows()[0]['row_id']),('ema_model',None),('epoch',30),('run_id','weighted_or_LRaw')]:
  z=copy.deepcopy(cp);z[key]=value;bad(lambda z=z:verify(z))
 expected=dict(checkpoint_sources=['parent/final_ssdg.pth'],parent_source='parent');done=dict(config=expected,epoch=10,optimizer_steps=2220,target_access=False);frozen=dict(config=expected,selection='final_only',target_access=False);init=dict(target_contact=False,checkpoint_sources=expected['checkpoint_sources'],scratch_only=False,ancestors=['parent']);contract=dict(role_ids={'L':['fixed']});check_source_metadata(done,frozen,init,contract,expected,contract['role_ids']);z=copy.deepcopy(init);z['ancestors']=['unknown'];bad(lambda:check_source_metadata(done,frozen,z,contract,expected,contract['role_ids']))
 text=Path(__file__).with_name('model.py').read_text(encoding='utf8');tree=ast.parse(text);actual=set();fieldsets=dict(CONCAT_FIELDS=CONCAT_FIELDS,BANK_FIELDS=BANK_FIELDS)
 for node in ast.walk(tree):
  if isinstance(node,ast.DictComp) and isinstance(node.key,ast.BinOp) and isinstance(node.key.left,ast.Constant) and isinstance(node.generators[0].iter,ast.Name):actual.update(node.key.left.value+k for k in fieldsets[node.generators[0].iter.id])
 assert actual==set(INFO_KEYS) and len(actual)==15 and 'shape/RD' not in actual
 assert text.count('capture_forward(self,')==3 and text.count('self.forward_identity_only(view,')==2
 source=Path(__file__).with_name('source.py').read_text(encoding='utf8');assert "r['row_id']==d.REFERENCE_ROW" in source and 'row=d.REFERENCE_ROW' in source
 bank_source=Path(__file__).with_name('bank.py').read_text(encoding='utf8');assert "ctx['train_loader'].dataset" in bank_source and 'for i in range(6300)' in bank_source and 'channel.apply(' in bank_source and 'torch.random.fork_rng(devices=[])' in bank_source
 from . import report
 assert [a for a,b in report.logical_rows()]==['NOAS30','AS30','BANK_IID30','BANK_STRAT30'] and len(report.PAIRS)==6
 for f in Path(__file__).parent.glob('*.py'):ast.parse(f.read_text(encoding='utf8'),filename=str(f))
 # Actual source-only summary JSON uses native bool/float for the three fixed criteria.
 old_rows,old_read,old_write,old_load=report.logical_rows,report.d.read,report.d.write,report.np.load
 class Data:
  def __init__(self,physics=False):
   y=np.arange(1800)%6;rx=np.asarray(d.RX)[np.arange(1800)%5];clean=np.eye(6,dtype=np.float32)[y]
   self.value={'signal_to_added_noise_db':np.broadcast_to(np.asarray([0.,5.,10.,20.])[np.arange(1800)%4],(2,3,11,1800)).copy()} if physics else dict(ids=np.asarray(['mock'+str(i) for i in range(1800)]),TX=y,RX=rx,clean=clean,satellite=np.tile(clean,(2,3,1,1)))
  def __enter__(self):return self.value
  def __exit__(self,*unused):pass
 def read_mock(path):
  name=Path(path).name
  if name=='completion.json':return dict(target_access=False,final_source_metrics=dict(source_val_count=27000,source_val_accuracy=.98,source_val_rx_accuracy={str(r):.98 for r in d.RX}))
  if name in ('source_frozen.json','complete.json'):return dict(target_access=False)
  return dict(role_ids={'L':['mock']})
 captured=[]
 try:
  report.logical_rows=lambda:[(a,Path('SYNTHETIC')/a) for a in ('NOAS30','AS30','BANK_IID30','BANK_STRAT30')];report.d.read=read_mock;report.d.write=lambda p,v:captured.append(v);report.np.load=lambda path,*args,**kw:Data(Path(path).name=='physics.npz')
  summary=report.source_decision();json.dumps(summary,allow_nan=False);assert type(summary['promotion']['passed']) is bool and not summary['promotion']['passed'] and len(summary['promotion']['conditions'])==3 and summary['scientific_failure_skips_formal'] is False and len(captured)==1
 finally:report.logical_rows,report.d.read,report.d.write,report.np.load=old_rows,old_read,old_write,old_load
 assert bank_source.index('if d.BANK.exists():raise FileExistsError')<bank_source.index('d.BANK.mkdir()') and bank_source.index("if (counts<=0).any():raise ValueError")<bank_source.index("d.write(d.BANK/'complete.json'")
 return dict(status='PASS',source_summary_native_JSON=True,empty_bank_bin_is_fatal_no_overwrite=True,protocol_negatives=negative,finite_risk=finite_risk_checks(),sampling=sampling_checks(),actual_target_boundary=actual_target_boundary(),Torch= torch_checks(),legacy_smoothing=legacy_smoothing_source_check(),emitted_native_info_keys=15,source_shared_bank_252000=True,one_parent_B01_two_new_finalIDs=True,real_model_checkpoint_smoke='launcher first770NN pending',NN=0,CP=0,IQ=0,updates=0,target=False)
if __name__=='__main__':print(json.dumps(run()))
