import unittest
import numpy as np
import torch
from ops import conjugate_iq,invariant_summary
from metrics import decide,arrays
class Checks(unittest.TestCase):
 def fixture(self):
  y=np.tile(np.arange(6),5);rx=np.repeat([1,3,4,6,8],6);z=np.eye(6)[y]*10;return y,z,z.copy(),rx
 def test_same_packet_exact_inverse_no_mutation(self):
  x=torch.randn(7,2,256);b=x.clone();z=conjugate_iq(x);self.assertTrue(torch.equal(x,b));self.assertTrue(torch.equal(conjugate_iq(z),x));self.assertTrue(torch.equal(x[:,0],z[:,0]));self.assertTrue(torch.equal(x[:,1],-z[:,1]));self.assertTrue(invariant_summary(x,z)['sample_magnitude_exact'])
 def test_fourier_reflection_not_phase_rotation(self):
  x=torch.randn(3,2,256);z=torch.complex(x[:,0],x[:,1]);q=conjugate_iq(x);q=torch.complex(q[:,0],q[:,1]);mirror=torch.arange(256).neg().remainder(256);self.assertTrue(torch.allclose(torch.fft.fft(q),torch.fft.fft(z)[:,mirror].conj(),atol=1e-5,rtol=1e-5))
 def test_bad_shape_dtype_nonfinite_rejected(self):
  for x in [torch.zeros(2,256),torch.zeros(1,3,256),torch.zeros(0,2,256),torch.zeros(1,2,256,dtype=torch.long),torch.full((1,2,256),float('nan'))]:
   with self.assertRaises(ValueError):conjugate_iq(x)
 def test_identity_pass_does_not_claim_Ag(self):
  d=decide(*self.fixture());self.assertTrue(d['candidate_admissible']);self.assertFalse(d['Ag_proved']);self.assertFalse(d['automatic_training']);self.assertFalse(d['target_access'])
 def test_receiver_loss_and_cohort_loss_fail(self):
  y,z,v,rx=self.fixture();v[0]=np.roll(v[0],1);d=decide(y,z,v,rx);self.assertFalse(d['candidate_admissible']);self.assertFalse(d['checks']['every_RX_accuracy_preserved']);self.assertFalse(d['checks']['fixed_raw_correct_retained'])
 def test_hidden_U_unseen_RX_bad_logits_rejected(self):
  y,z,v,rx=self.fixture()
  for yy,rr,vv in [(np.full_like(y,-1),rx,v),(y,np.full_like(rx,99),v),(y,rx,np.full_like(v,np.nan)),(y,rx,v[:,:5])]:
   with self.assertRaises(ValueError):arrays(yy,z,vv,rr)
if __name__=='__main__':unittest.main()
