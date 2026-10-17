"""Focused finite-IQ timing/routing/source protocol regressions,zero real IQ/CP."""
import random,unittest
import torch
from .timing import transform,warp
from .route import TimingRoute
class Cases(unittest.TestCase):
 def meta(self):return {k:torch.full((128,),v,dtype=torch.long) for k,v in [('eq_i',1),('rx_i',3),('day_i',2)]}|{'sig_i':torch.arange(128),'base_index':torch.arange(128)}
 def test_reflect_without_wrap_and_identity(self):
  x=torch.arange(256,dtype=torch.float32)[None,None,:].repeat(3,2,1);out,info=warp(x,[-16,0,16]);self.assertTrue(torch.equal(out[1],x[1]));self.assertEqual(info['indices_min'],0);self.assertEqual(info['indices_max'],255)
  # Same scalar normalization in every coordinate;positive delay takes prefix from16 rather than240.
  self.assertAlmostEqual(float(out[2,0,0]/out[2,0,16+16]),1.,places=6);self.assertAlmostEqual(float(out[0,0,255]/out[0,0,223]),1.,places=6)
  p=x.double().square().mean((1,2));q=out.double().square().mean((1,2));self.assertLess(float(((p-q).abs()/p).max()),1e-6)
 def test_private_refresh_and_batch_order(self):
  x=torch.randn(6,2,256,generator=torch.Generator().manual_seed(8));ids=list(map(str,range(6)));prng=random.getstate();trng=torch.random.get_rng_state().clone()
  a,ma=transform(x,ids,'trainL/e1/s1',seed=202610174101);b,mb=transform(x.flip(0),ids[::-1],'trainL/e1/s1',seed=202610174101);self.assertTrue(torch.equal(a,b.flip(0)));self.assertEqual(ma['offset_samples'],mb['offset_samples'][::-1]);self.assertEqual(prng,random.getstate());self.assertTrue(torch.equal(trng,torch.random.get_rng_state()))
  _,mc=transform(x,ids,'trainL/e1/s2',seed=202610174101);self.assertNotEqual(ma['offset_samples'],mc['offset_samples'])
 def test_route_and_hidden_labels(self):
  raw=torch.randn(128,2,256,generator=torch.Generator().manual_seed(9));ordinary=raw+1;meta=self.meta();a,info=TimingRoute(True).apply(raw,ordinary,meta,1);b,_=TimingRoute(True).apply(raw,ordinary,dict(meta,TX=torch.arange(128)%6),1);self.assertTrue(torch.equal(a,b));self.assertFalse(torch.equal(a,ordinary));same,_=TimingRoute(False).apply(raw,ordinary,meta,1);self.assertTrue(torch.equal(same,raw))
 def test_illegal_metadata_and_step(self):
  x=torch.ones(128,2,256);m=self.meta()
  for xx,mm,s in [(x[:64],m,1),(x,{**m,'rx_i':torch.zeros(128,dtype=torch.long)},1),(x,{k:v[:127] for k,v in m.items()},1),(x,m,2)]:
   with self.assertRaises(ValueError):TimingRoute(True).apply(xx,x,mm,s)
  for off in ([17]*128,[-17]*128):
   with self.assertRaises(ValueError):warp(x,off)
 def test_repeated_ids_retained_and_reference_validity(self):
  x=torch.ones(128,2,256);m=self.meta();m={k:v*0+v[0] for k,v in m.items()};r=TimingRoute(True);out,info=r.apply(x,x,m,1);self.assertEqual(len(set(info['ids'])),128);self.assertEqual(len(set(info['physical_ids'])),1);self.assertEqual(r.packets,128);self.assertTrue(torch.equal(out,x));self.assertEqual(info['meta']['full_As_reference_raw_dependency'],[3,176])
 def test_native_argument_nan_placeholders(self):
  from .matching import argument_differences
  a={'placeholder':float('nan'),'real':.2,'run_id':'new'};b={'placeholder':float('nan'),'real':.2,'run_id':'old'}
  self.assertEqual(argument_differences(a,b,{'run_id'}),[]);self.assertTrue(a['placeholder']!=a['placeholder'])
  self.assertEqual(argument_differences(a,{**b,'placeholder':0.},{'run_id'}),['placeholder'])
  self.assertEqual(argument_differences(a,{**b,'real':.3},{'run_id'}),['real'])
  self.assertEqual(argument_differences({'missing':None},{},set()),['missing'])
 def test_fixed_source_gain_conditions(self):
  from .criteria import decide
  cv=dict(B00=.98,B01=.977,B10=.98,B11=.98);a=decide(cv,[.5,.5],[.55,.55],[[.6,.6],[.6,.6]],[[.61,.61],[.61,.61]]);self.assertTrue(a['source_replication_evidence'])
  cv['B10']=.9774;b=decide(cv,[.5,.5],[.55,.55],[[.6,.6],[.6,.6]],[[.61,.61],[.61,.61]]);self.assertFalse(b['checks']['Ag_clean_V_preserved'])
  with self.assertRaises(ValueError):decide(cv,[.5],[.6],[[.6,.6],[.6,.6]],[[.61,.61],[.61,.61]])
if __name__=='__main__':unittest.main()
