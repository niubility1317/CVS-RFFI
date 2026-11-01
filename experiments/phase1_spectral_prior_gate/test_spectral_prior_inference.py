import ast,unittest,torch
from pathlib import Path
from torch.nn import functional as F
from spectral_prior_gate import SpectralPriorGate
try:import spectral_prior_inference as inference
except ImportError:inference=None
class FixedPhase:
 def anchor(self,x):return x.roll(3,-1)
class FixedGate:
 def __call__(self,x):return torch.tensor([-10.,10.],device=x.device)[:len(x)]
class ClosedGate:
 def __call__(self,x):return torch.full((len(x),),-10.,device=x.device)
class NeverPhase:
 def anchor(self,x):raise AssertionError('disabled records must bypass restoration')
class InferenceTests(unittest.TestCase):
 def setUp(self):
  self.assertIsNotNone(inference,'inference implementation required');torch.manual_seed(13);self.bank=torch.randn(32,256,dtype=torch.complex64);c=self.bank[:,64:-64];self.power=c.abs().square().sum(1).clamp_min(1e-12);self.unit=c/self.power.sqrt()[:,None];self.net=SpectralPriorGate().eval()
  with torch.no_grad():self.net.output.bias.copy_(torch.tensor([.4,.1]))
 def test_disabled_records_are_exactly_preserved(self):
  x=torch.randn(2,2,256);out,enabled=inference.apply(self.net,FixedPhase(),FixedGate(),self.bank,self.power,self.unit,.5,.8,x)
  self.assertFalse(enabled[0]);self.assertTrue(enabled[1]);self.assertTrue(torch.equal(out[0],x[0]));a=FixedPhase().anchor(x[1:]);p=inference.make_prior(a,self.bank,self.power,self.unit);expected=a+.5*(self.net(a,p)-a);self.assertTrue(torch.allclose(out[1:],expected,atol=1e-6,rtol=1e-6))
 def test_all_disabled_skips_restoration(self):
  x=torch.randn(3,2,256);out,enabled=inference.apply(self.net,NeverPhase(),ClosedGate(),self.bank,self.power,self.unit,1.,.8,x);self.assertTrue(torch.equal(out,x));self.assertFalse(enabled.any())
 def test_retrieval_matches_predeclared_source_selection(self):
  src=Path(__file__).with_name('spectral_prior_source.py');tree=ast.parse(src.read_text(encoding='utf8'));fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='make_prior');env={'torch':torch,'F':F};exec(compile(ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[])),str(src),'exec'),env)
  x=torch.randn(3,2,256);expected=env['make_prior'](x,self.bank,self.power,self.unit);actual=inference.make_prior(x,self.bank,self.power,self.unit);self.assertTrue(torch.equal(actual,expected))
if __name__=='__main__':unittest.main()
