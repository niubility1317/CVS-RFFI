"""Fixed-scope negatives, analytic physical-coordinate projection, and full F synthetic hook."""
import copy,json,torch
from types import SimpleNamespace
from . import design as d
from .core import geometry,capture,consume
from .checkpoint_policy import load

def run(full=True):
 from .registration import expand_paths
 probe=dict(mode='PROJECT',row_id='PROJECT-s2026092701',cost='MEASURE/PROJECT twoVJP',path='PROJECT/runs/x',nested=['PROJECT','PROJECT/releases/x'])
 result=expand_paths(probe,'/example/project')
 assert result==dict(mode='PROJECT',row_id='PROJECT-s2026092701',cost='MEASURE/PROJECT twoVJP',path='/example/project/runs/x',nested=['PROJECT','/example/project/releases/x'])
 assert probe['path']=='PROJECT/runs/x'
 negative=0
 def bad(f):
  nonlocal negative
  try:f()
  except (ValueError,TypeError,KeyError,FloatingPointError):negative+=1
  else:raise AssertionError('Invalid fixed scope accepted')
 c=d.config(d.rows()[0])
 for k,v in [('epochs',11),('updates',2442),('cumulative_epochs',31),('parent_source','other'),('new_losses',1),('new_U_rules',1),('new_neural_forwards',1),('model_seed',2026092702),('As',False),('new_Ag',True),('gradient_mode','other'),('batch_mode','RANDOM')]:
  a=copy.deepcopy(c);a[k]=v;bad(lambda a=a:d.validate(a))
 for k,v in [('RX',[3,4,6,8]),('L',5580),('U_TX_truth_hidden',False),('source_RX_holdout',1)]:
  a=copy.deepcopy(c);a['data_contract'][k]=v;bad(lambda a=a:d.validate(a))
 sentinel=object();calls=[]
 def actual(path,dev):calls.append(path);return 'FINAL'
 assert load('/p/final','cpu','/p/final','/r/final',sentinel,actual) is sentinel and not calls
 assert load('/r/final','cpu','/p/final','/r/final',sentinel,actual)=='FINAL'
 for path in ('/other/final','/r/latest','/p/best'):bad(lambda path=path:load(path,'cpu','/p/final','/r/final',sentinel,actual))
 state=torch.random.get_rng_state().clone();p=torch.nn.Parameter(torch.zeros(2,dtype=torch.float64));named=(('ordinary',p),)
 g=geometry(named,(torch.tensor([1.,0.]),),(torch.tensor([-2.,3.]),));assert g['coefficient']==2. and g['projected_satellite_dot']==0.
 assert geometry(named,(torch.zeros(2),),(torch.tensor([1.,2.]),))['coefficient']==0.
 assert geometry(named,(torch.tensor([1.,0.]),),(torch.tensor([2.,3.]),))['coefficient']==0.
 from experiments.cvs_rdrop_sinc_units.optimizer import QNAMES
 named=((QNAMES[0],p),('body',p))
 gc=(torch.tensor([25000.,0.]),torch.tensor([1.,0.]));gs=(torch.tensor([-50000.,0.]),torch.tensor([0.,3.]))
 gg=geometry(named,gc,gs);assert gg['coefficient']==1. and gg['dot']==-2.
 bad(lambda:geometry((('p',p),),(torch.full((2,),float('nan')),),(torch.zeros(2),)))
 bad(lambda:geometry((('p',p),),(torch.zeros(3),),(torch.zeros(2),)))
 class Toy(torch.nn.Module):
  def __init__(self,mode):
   super().__init__();self.p=torch.nn.Parameter(torch.zeros(2,dtype=torch.float64));self._pc_mode=mode;self._pc_pending=None;self._pc_clips=0;self._pc_vjp_calls=0;self._pc_records=[]
 outputs={}
 for mode in d.MODES:
  m=Toy(mode);lo=m.p[0];ls=-2*m.p[0]+3*m.p[1];aux=7*m.p[0]-4*m.p[1]
  capture(m,lo,ls,1);(lo+ls+aux).backward();consume(m,m.parameters(),5.)
  expected=torch.tensor([8.,-1.] if mode=='PROJECT' else [6.,-1.],dtype=torch.float64)
  assert torch.equal(m.p.grad,expected) and m._pc_pending is None and m._pc_clips==1 and m._pc_vjp_calls==(0 if mode=='ORIGINAL' else 2)
  outputs[mode]=m.p.grad.tolist()
 assert torch.equal(state,torch.random.get_rng_state())
 synthetic=0;maxerror=0.
 if full:
  from . import model as api
  from experiments.cvs_rdrop_sinc_units.optimizer import build,clip_physical
  from experiments.cvs_fullsource_concat20 import supcon
  class FakeChannel:
   def apply(self,x,meta,**kw):return x*.9, [dict(scene='practical_mid_urban',state_end='post_sync',receiver_lock_status='ok') for _ in x]
  original_schedule=supcon.schedule;supcon.schedule=lambda epoch:.1
  try:
   torch.set_num_threads(2);torch.manual_seed(20261018201);raw=torch.randn(128,2,256);ordinary=raw*.7+.13;y=torch.arange(128)%6;domains=torch.arange(128)%15;rx=torch.tensor(supcon.RX_BY_DOMAIN)[domains]
   initial=None;reference=None;rng=None
   for row in d.rows():
    conf=d.config(row);m=api.build(conf,'cpu')
    if initial is None:initial=copy.deepcopy(m.state_dict())
    else:m.load_state_dict(initial,strict=True)
    m._full_channel=FakeChannel();m.train();opt=build(m,list(m.parameters()),'CENTER_UNITS',lr=.0002,eps=1e-8,weight_decay=.0001,foreach=False)
    torch.manual_seed(20261018202)
    with m._phase1_labeled_feature_context():out=m(ordinary,y_tx=y,return_aux=True)
    ce=torch.nn.functional.cross_entropy(out['tx_logits'],y,label_smoothing=.01)
    correction,metrics=m._source_labeled_feature_hook(model=m,labeled_output=out,labels=y,extra=(None,dict(rx_i=rx)),domains=domains,optimizer_step=1,identity_loss=ce,labeled_count=128,labeled_inputs=raw)
    (ce+correction).backward();actual_rng=torch.random.get_rng_state().clone();params=consume(m,m.parameters(),5.);clip_physical(opt,params,5.);grads={n:p.grad.detach().clone() for n,p in m.named_parameters() if p.grad is not None}
    if row['gradient_mode']=='ORIGINAL':reference=grads;rng=actual_rng
    elif row['gradient_mode']=='MEASURE':
     assert set(grads)==set(reference) and torch.equal(actual_rng,rng)
     maxerror=max(float((v-reference[n]).abs().max()) for n,v in grads.items());assert maxerror<=1e-6
    else:assert torch.equal(actual_rng,rng)
    opt.step();teacher=copy.deepcopy(m).eval();assert teacher._pc_pending is None and teacher._pc_clips==1 and teacher._source_labeled_feature_hook.__self__ is teacher
    assert m._full_L_packets==512 and m._pc_vjp_calls==(0 if row['gradient_mode']=='ORIGINAL' else 2) and opt._sinc_unit_steps==1
    synthetic+=512
  finally:supcon.schedule=original_schedule
 return dict(status='PASS',protocol_negatives=negative,analytic_gradient_results=outputs,physical_Sinc_metric=True,remaining_aux_gradient_preserved=True,global_RNG_preserved_for_core=True,synthetic_F_forward_packets=synthetic,ORIGINAL_MEASURE_gradient_max_error=maxerror,IQ=0,CP=0,real_updates=0,synthetic_updates=3 if full else 0)
if __name__=='__main__':print(json.dumps(run()))
