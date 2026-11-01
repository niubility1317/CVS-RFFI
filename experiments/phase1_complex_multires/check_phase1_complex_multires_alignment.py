import ast
from pathlib import Path
import numpy as np

source = Path(__file__).with_name('phase1_complex_multires_source.py').read_text(encoding='utf8')
tree = ast.parse(source)
func = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'align_to_reference')
namespace = {'np': np}
exec(compile(ast.Module(body=[func], type_ignores=[]), '<alignment-test>', 'exec'), namespace)
align = namespace['align_to_reference']
rng = np.random.default_rng(1234)
reference = rng.normal(size=256) + 1j*rng.normal(size=256)
wave = np.stack((np.roll(reference, 61)*(2+.3j), np.roll(reference, -61)*(.5-.8j)))
_, error = align(wave, np.stack((reference, reference)))
assert np.max(error) < 1e-20
print('VERIFIED source alignment at both ±61 sample shifts')

import torch
from phase1_complex_multires_mask import aligned_observation
rng = np.random.default_rng(12345)
anchor_np = rng.normal(size=(12, 2, 256)).astype(np.float32)
received_np = (anchor_np + rng.normal(scale=.4, size=anchor_np.shape)).astype(np.float32)
a = anchor_np[:,0].astype(np.float64)+1j*anchor_np[:,1]
r = received_np[:,0].astype(np.float64)+1j*received_np[:,1]
expected, _ = align(r, a)
_, residual = align(a, r)
power = np.mean(abs(r[:,64:-64])**2,axis=1).clip(min=1e-20)
fraction = np.clip(residual/power,1e-8,1.)
proxy = 10*np.log10(np.maximum(1-fraction,1e-8)/fraction)
ratio = np.sqrt(np.sum((received_np-anchor_np)**2,axis=(1,2))/np.maximum(np.sum(received_np**2,axis=(1,2)),1e-12))
expected_gate = (ratio>.2)&(proxy<10.)
actual, actual_gate = aligned_observation(torch.tensor(anchor_np),torch.tensor(received_np))
actual_complex = actual[:,0].numpy()+1j*actual[:,1].numpy()
assert np.max(abs(actual_complex-expected)) < 1e-4
assert np.array_equal(actual_gate.numpy(),expected_gate)
print('VERIFIED vectorized alignment and gate against source implementation')
