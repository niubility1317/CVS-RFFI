import importlib.util,unittest
from pathlib import Path
from types import SimpleNamespace
import torch
import torch.nn.functional as F
ROOT=Path(__file__).parent
for name in ('losses','criteria','protocol'):
 s=importlib.util.spec_from_file_location('hard45_'+name,ROOT/(name+'.py'));m=importlib.util.module_from_spec(s);s.loader.exec_module(m);globals()[name]=m

def original(a,b,y,alpha,smoothing=.01):
 if smoothing!=.01 or a.shape!=b.shape or a.shape!=(len(y),6) or y.dtype!=torch.long or not bool(((y>=0)&(y<6)).all()) or not torch.isfinite(a).all() or not torch.isfinite(b).all() or alpha not in (0.,1.):raise ValueError('Invalid original objective')
 c1=F.cross_entropy(a,y,label_smoothing=smoothing);c2=F.cross_entropy(b,y,label_smoothing=smoothing);p,q=a.log_softmax(1),b.log_softmax(1);kl=.5*((p.exp()-q.exp())*(p-q)).sum(1).mean();return .5*(c1+c2)+alpha*kl,.5*(c1+c2),kl,c1,c2
class Checks(unittest.TestCase):
 def tensors(self):
  g=torch.Generator().manual_seed(45);return torch.randn(8,6,dtype=torch.float64,generator=g,requires_grad=True),torch.randn(8,6,dtype=torch.float64,generator=g,requires_grad=True),torch.arange(8)%6
 def test_exact_hard_ce_KL_and_gradient(self):
  a,b,y=self.tensors();mod=SimpleNamespace(terms=original);old=original(a,b,y,1.);records=[]
  with losses.hard_L_terms(mod,records):new=mod.terms(a,b,y,1.)
  self.assertIs(mod.terms,original);self.assertEqual(len(records),1);self.assertTrue(torch.equal(new[2],old[2]));expected=.5*(F.cross_entropy(a,y)+F.cross_entropy(b,y));self.assertTrue(torch.equal(new[1],expected))
  ga=torch.autograd.grad(new[0]-old[0],a)[0];expected_delta=.005*(torch.full_like(a,1/6)-F.one_hot(y,6))/len(y);self.assertTrue(torch.allclose(ga,expected_delta,atol=1e-14,rtol=1e-12))
 def test_scope_restores_before_U_or_exception(self):
  a,b,y=self.tensors();mod=SimpleNamespace(terms=original);old=mod.terms(a,b,y,1.)[0]
  with self.assertRaises(RuntimeError):
   with losses.hard_L_terms(mod,[]):raise RuntimeError('Abort L')
  self.assertIs(mod.terms,original);self.assertTrue(torch.equal(old,mod.terms(a,b,y,1.)[0]))
 def test_no_RNG_or_input_mutation(self):
  a,b,y=self.tensors();snap=(a.detach().clone(),b.detach().clone(),y.clone());rng=torch.random.get_rng_state().clone();mod=SimpleNamespace(terms=original)
  with losses.hard_L_terms(mod,[]):mod.terms(a,b,y,1.)
  self.assertTrue(torch.equal(rng,torch.random.get_rng_state()));self.assertTrue(all(torch.equal(x,z) for x,z in zip((a,b,y),snap)))
 def test_invalid_inputs_restored(self):
  a,b,y=self.tensors();bad=[(a,b,y,2.),(a,b,y,1.,0.),(a,b,torch.full_like(y,6),1.),(a,b[:2],y,1.),(a*float('nan'),b,y,1.)]
  for args in bad:
   mod=SimpleNamespace(terms=original)
   with self.assertRaises(ValueError):
    with losses.hard_L_terms(mod,[]):mod.terms(*args)
   self.assertIs(mod.terms,original)
 def metrics(self):
  c=dict(B00=.8,H00=.803,B01=.8,H01=.803);u={a:[[.8,.6,.5],[.8,.6,.5]] for a in criteria.ARMS};u['H01']=[[.8,.61,.51],[.8,.61,.51]];r={a:{str(x):.8 for x in (1,3,4,6,8)} for a in criteria.ARMS};return c,u,r
 def test_source_conditions_no_goalclaim(self):
  v=criteria.decide(*self.metrics());self.assertTrue(v['source_qualified']);self.assertFalse(v['Ag_proved']);self.assertFalse(v['goal_complete'])
  c,u,r=self.metrics();u['H01'][1][2]=.499;self.assertFalse(criteria.decide(c,u,r)['source_qualified'])
 def test_source_incomplete_nonfinite_refused(self):
  c,u,r=self.metrics();c.pop('B01')
  with self.assertRaises(ValueError):criteria.decide(c,u,r)
  c,u,r=self.metrics();u['H01'][0][1]=float('nan')
  with self.assertRaises(ValueError):criteria.decide(c,u,r)
 def test_checkpoint_boundary(self):
  self.assertEqual(protocol.loader_kind('/a/final','/a/parent','/a/final'),'final')
  with self.assertRaises(ValueError):protocol.loader_kind('/b/foreign','/a/parent','/a/final')
if __name__=='__main__':unittest.main()
