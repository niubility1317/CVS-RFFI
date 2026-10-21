import unittest
import torch
from .runtime import native_source_logits

class VersionedNative(torch.nn.Module):
 def forward_identity_only(self,x):
  # Native centered-Sinc constructs p0+s*q and reads its version for caching.
  physical=x+1
  _=physical._version
  return {'tx_logits':physical}

class NativeSmokeTests(unittest.TestCase):
 def test_native_version_cache_works_inside_outer_inference(self):
  x=torch.zeros(2,6)
  with torch.inference_mode():
   y=native_source_logits(VersionedNative(),x)
  self.assertTrue(torch.equal(y,torch.ones(2,6)))
  self.assertFalse(y.requires_grad)

if __name__=='__main__':unittest.main()
