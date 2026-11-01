import torch
from phase1_complex_multires_mask import ComplexMultiResolutionMask, aligned_observation

torch.manual_seed(1201)
model = ComplexMultiResolutionMask()
a = torch.randn(3, 2, 256)
r = a + 0.2 * torch.randn_like(a)
with torch.no_grad():
    initial = model(a, r, torch.ones(3))
assert torch.allclose(initial, a, atol=1e-5, rtol=1e-5)
with torch.no_grad():
    model.mask_net[-1].weight.normal_(0, 0.01)
    gated = model(a, r, torch.zeros(3))
assert torch.equal(gated, a)
output = model(a, r, torch.ones(3))
loss = (output - r).square().mean()
loss.backward()
assert torch.isfinite(output).all()
assert model.mask_net[-1].weight.grad is not None
assert torch.isfinite(model.mask_net[-1].weight.grad).all()
assert model.mask_net[-1].weight.grad.abs().sum().item() > 0
try:
    model(a[:, :, :-1], r[:, :, :-1], torch.ones(3))
except ValueError:
    pass
else:
    raise AssertionError('bad shape must fail')
print('VERIFIED multi-resolution complex mask identity, gate and gradient')

# Synthetic 61-sample circular delay plus complex gain must align over the
# central interval, even though the unused edge samples differ.
torch.manual_seed(61)
ref = torch.randn(2, 2, 256)
z = torch.complex(ref[:,0],ref[:,1])
gain = torch.tensor([2+.3j, .5-.8j])[:,None]
wave = torch.roll(z,61,-1)*gain
received = torch.stack((wave.real,wave.imag),1)
aligned, gate = aligned_observation(ref,received)
assert torch.allclose(aligned[:,:,64:-64],ref[:,:,64:-64],atol=2e-5,rtol=2e-5)
assert gate.shape == (2,) and gate.dtype == torch.bool
print('VERIFIED vectorized received-IQ alignment')
