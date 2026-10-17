"""Focused clean-retaining routing, all labeled As, no extra data or RNG."""
import unittest
from types import SimpleNamespace
import torch
from .route import mask_for_step,subset_meta,compose
class Cases(unittest.TestCase):
 def test_mask_count_refresh_no_global_rng(self):
  state=torch.random.get_rng_state().clone();a=mask_for_step(1,'cpu');b=mask_for_step(2,'cpu')
  self.assertEqual(int(a.sum()),64);self.assertTrue(torch.equal(a,~b));self.assertTrue(torch.equal(state,torch.random.get_rng_state()))
 def test_only_physical_source_metadata(self):
  m={k:torch.full((128,),v,dtype=torch.long) for k,v in [('eq_i',1),('rx_i',3),('day_i',2),('sig_i',1),('base_index',2)]};m['TX']=torch.zeros(128,dtype=torch.long)
  out=subset_meta(m,mask_for_step(1,'cpu'));self.assertNotIn('TX',out);self.assertTrue(all(len(x)==64 for x in out.values()))
  with self.assertRaises(ValueError):subset_meta({**m,'rx_i':torch.zeros(128,dtype=torch.long)},mask_for_step(1,'cpu'))
 def test_all_generated_satellite_used_and_primary_preserved(self):
  x=torch.randn(128,2,256,generator=torch.Generator().manual_seed(7));y=torch.arange(128)%6;d=torch.arange(128)%15;mask=mask_for_step(1,'cpu');sat=torch.ones(64,2,256)*9
  batch=SimpleNamespace(x=torch.cat((x[mask],sat)),y=torch.cat((y[mask],y[mask])),d_raw=torch.cat((d[mask],d[mask])),clean_batch_size=64,total_batch_size=128,nuisance=torch.arange(128)[:,None].float(),nuisance_valid=torch.arange(128)>=64)
  result=compose(batch,x,y,d,mask)
  self.assertTrue(torch.equal(result.x[:128],x));self.assertTrue(torch.equal(result.x[128:][~mask],x[~mask]));self.assertTrue(torch.equal(result.x[128:][mask],sat));self.assertTrue(torch.equal(result.y,torch.cat((y,y))));self.assertTrue(torch.equal(result.d_raw,torch.cat((d,d))));self.assertEqual(int(result.nuisance_valid.sum()),64);self.assertTrue(torch.equal(result.nuisance[128:][mask],torch.arange(64,128)[:,None].float()));self.assertEqual((result.clean_batch_size,result.total_batch_size),(128,256))
 def test_wrong_labels_step_batch_and_nonfinite_rejected(self):
  for step in (0,6661,True):
   with self.assertRaises(ValueError):mask_for_step(step,'cpu')
  x=torch.ones(128,2,256);y=torch.arange(128)%6;d=y;mask=mask_for_step(1,'cpu');batch=SimpleNamespace(x=torch.ones(128,2,256),y=torch.zeros(128,dtype=torch.long),d_raw=torch.cat((d[mask],d[mask])),nuisance=None,nuisance_valid=None)
  with self.assertRaises(ValueError):compose(batch,x,y,d,mask)
  batch.y=torch.cat((y[mask],y[mask]));batch.x[0,0,0]=float('nan')
  with self.assertRaises(ValueError):compose(batch,x,y,d,mask)
 def test_supervised_ce_all_samples_weight(self):
  import torch.nn.functional as F
  a=torch.randn(256,6,generator=torch.Generator().manual_seed(8),requires_grad=True);y=torch.arange(128)%6;mask=mask_for_step(1,'cpu')
  loss=.5*F.cross_entropy(a[:128],y,label_smoothing=.01)+.5*F.cross_entropy(a[128:],y,label_smoothing=.01);loss.backward()
  self.assertTrue(bool((a.grad.norm(dim=1)>0).all()));self.assertAlmostEqual(.5+.5*float((~mask).float().mean()),.75);self.assertAlmostEqual(.5*float(mask.float().mean()),.25)
 def test_synthetic_native_execution_output_is_isolated(self):
  # Native practical adapter writes first-batch metadata under args.output_dir.
  import ast
  from pathlib import Path
  tree=ast.parse((Path(__file__).parent/'synthetic.py').read_text(encoding='utf8'));assigns=[n for n in ast.walk(tree) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Attribute) and isinstance(t.value,ast.Name) and t.value.id=='a' and t.attr=='output_dir' for t in n.targets)]
  self.assertEqual(len(assigns),1);self.assertIn("native_execution",ast.unparse(assigns[0].value));self.assertIn("Synthetic check must never create real training output",(Path(__file__).parent/'synthetic.py').read_text(encoding='utf8'))
if __name__=='__main__':unittest.main()
