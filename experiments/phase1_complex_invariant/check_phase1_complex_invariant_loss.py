import torch
from phase1_complex_invariant_loss import complex_gain_invariant_error

# A source-matched target plus orthogonal nuisance. Shrinking both signal and
# nuisance lowers the old fixed-target IQ loss, despite worsening shape.
signal=torch.ones(1,2,256);signal[:,1]=0
noise=torch.ones(1,2,256);noise[:,0,1::2]=-1;noise[:,1]=0
anchor=signal+noise
candidate=.7*signal+.8*noise
old_anchor=(anchor-signal).square().mean();old_candidate=(candidate-signal).square().mean()
assert old_candidate<old_anchor
new_anchor=complex_gain_invariant_error(anchor,signal)
new_candidate=complex_gain_invariant_error(candidate,signal)
assert torch.all(new_candidate>new_anchor)
# Global amplitude and carrier phase do not alter this source IQ metric.
z=torch.complex(candidate[:,0],candidate[:,1])*(2.4-.7j)
rotated=torch.stack((z.real,z.imag),1)
torch.testing.assert_close(complex_gain_invariant_error(candidate,signal),complex_gain_invariant_error(rotated,signal),rtol=1e-6,atol=1e-6)
# Silent predictions have finite loss and finite gradients.
zero=torch.zeros_like(anchor,requires_grad=True)
loss=complex_gain_invariant_error(zero,signal).sum();loss.backward()
assert torch.isfinite(loss) and torch.isfinite(zero.grad).all()
assert abs(loss.item()-1.)<1e-6
print('VERIFIED fixed-target shrinkage counterexample, complex invariance and finite gradients')
