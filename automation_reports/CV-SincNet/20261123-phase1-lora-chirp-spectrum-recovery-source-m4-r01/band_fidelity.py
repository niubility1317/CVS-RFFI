import json
from pathlib import Path
import torch
from experiments.cric_dura_comparison.io import read
from experiments.pamf_lora_main.chirp_spectrum_source import RECIPES, recover
from experiments.pamf_lora_main.channel_blind_source import setup, proof
from experiments.pamf_lora_main.dura_repair import PacketRestorer, complex_iq, real_iq
from experiments.pamf_lora_main.joint_unroll_source import coordinates
from experiments.pamf_lora_main.pilot_joint import generic_distortion

base = Path('/home/szu2070436088/2510044040/CV-SincNet')
run = '20261123-phase1-lora-chirp-spectrum-recovery-source-m4-r01'
rt = read(base / 'releases' / (run+'-release') / 'runtime.json')
root = Path(rt['run_root'])
torch.set_num_threads(2)
_,_,held = setup(rt, 'cpu')[0]
contract = setup(rt, 'cpu', smoke=True)[1]
models = []
thresholds = []
for recipe in RECIPES.values():
    checkpoint = torch.load(root/f'{recipe.name}-model.pt', map_location='cpu', weights_only=True)
    assert checkpoint['proof'] == proof(contract, recipe)
    model = recipe.model().eval()
    model.load_state_dict(checkpoint['weights'], strict=True)
    models.append(model)
    thresholds.append(checkpoint['threshold'])
assert thresholds[0] == thresholds[1]
tracker = PacketRestorer('consensus').eval()
rng = torch.Generator().manual_seed(2026112309)
modes = ('direct','shared','separate','separate_half')
freq = torch.fft.fftfreq(256).abs()
bounds = ((0,8),(8,32),(32,64),(64,128))
masks = [(freq > a/256) & (freq <= b/256) for a,b in bounds]
totals = torch.zeros(len(modes), len(bounds), 5, dtype=torch.float64)
active = 0
with torch.no_grad():
    for source in held.split(16):
        y = real_iq(generic_distortion(complex_iq(source),rng))
        _, rotation, _, rms = coordinates(tracker,y)
        basis = rotation*rms
        variants, count = recover(y,tracker,models,thresholds[0],half=True)
        active += count
        expected = torch.fft.fft(complex_iq(source)/basis,norm='ortho')
        for idx, iq in enumerate((y,*variants)):
            current = torch.fft.fft(complex_iq(iq)/basis,norm='ortho')
            for k, band in enumerate(masks):
                p, t = current[...,band], expected[...,band]
                powers, power_t = p.abs().square(), t.abs().square()
                amp_error = (p.abs()-t.abs()).square().sum()
                err = (p-t).abs().square().sum()
                dot = (p*t.conj()).real.sum()
                totals[idx,k] += torch.stack((err,powers.sum(),power_t.sum(),dot,amp_error)).double()
result = {}
for mode, rows in zip(modes, totals):
    result[mode] = {}
    for k, ((lo, hi), info) in enumerate(zip(bounds, rows)):
        error, energy, truth, dot, amplitude = info.tolist()
        result[mode][f'{lo+1}-{hi} kHz'] = dict(
           normalized_complex_error=error/truth, normalized_amplitude_error=amplitude/truth,
           real_complex_cosine=dot/(energy*truth)**.5, error_vs_direct=error/float(totals[0,k,0]))
print(json.dumps(dict(records=len(held), trigger_count=active, fixed_U_only=True, independent_generic_seed=2026112309,
                      basis='original received carrier coordinates', routes_or_target_read=False,
                      metrics=result), indent=2, allow_nan=False))
