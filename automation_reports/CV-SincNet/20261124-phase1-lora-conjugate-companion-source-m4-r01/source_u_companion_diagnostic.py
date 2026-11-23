
import json
from pathlib import Path
import torch
from experiments.cric_dura_comparison.io import read, tensor
from experiments.pamf_lora_main.channel_blind_source import partition
from experiments.pamf_lora_main.data import SOURCE
from experiments.pamf_lora_main.dura_repair import PacketRestorer, complex_iq, real_iq
from experiments.pamf_lora_main.full_packet_publish import DATA_RUN
from experiments.pamf_lora_main.full_packet_views import source_contract, source_role
from experiments.pamf_lora_main.joint_unroll_source import coordinates
from experiments.pamf_lora_main.pilot_joint import apply_echo, echo_taps
from experiments.pamf_lora_main.rf_noise_transport import colored_observation
from experiments.pamf_lora_main.tail_band_ablation_source import BANDS
from experiments.pamf_unrolled_restore.denoise import complementary_context, spectral_companion
import numpy as np

torch.set_num_threads(2)
base=Path('/home/szu2070436088/2510044040/CV-SincNet')
release=base/'releases'/('20261124-phase1-lora-conjugate-companion-source-m4-r01'+'-release')
rt=read(release/'runtime.json')
root=Path(rt['source_root'])
contract=source_contract(root)
_, rx, _, _=source_role(root, contract, 'U_s')
_, _, ids=partition(rx)
data=np.load(base/'runs'/DATA_RUN/'source_views'/'U_s.npy', mmap_mode='r', allow_pickle=False)
clean=tensor(data[0, ids], 'cpu')
rng=torch.Generator(device='cpu').manual_seed(2026112415)
tracker=PacketRestorer('consensus').eval()
frequency=torch.fft.fftshift(torch.fft.fftfreq(256)).abs()*256
sums={k: {name: torch.zeros(len(BANDS), dtype=torch.float64) for name in
            ('mirror', 'adjacent', 'repeat_mirror', 'repeat_adjacent')}
      for k in ('clean', 'disturbed')}
n_rows=0
with torch.no_grad():
    for x in clean.split(20):
        z=complex_iq(x)
        perturbed=real_iq(colored_observation(apply_echo(z, echo_taps(z, rng)), rng))
        for name, y in (('clean', x), ('disturbed', perturbed)):
            frame, rotation, q, rms=coordinates(tracker, y)
            observed=torch.fft.fftshift(torch.fft.fft(frame, norm='ortho'), dim=-1)
            reference=torch.fft.fftshift(torch.fft.fft(z/(rotation*rms), norm='ortho'), dim=-1)
            others,_=complementary_context(observed,q)
            for label,sig in (('',observed),('repeat_',others)):
                for mode in ('mirror','adjacent'):
                    paired=spectral_companion(sig,mode)
                    for k,(lo,hi) in enumerate(BANDS):
                        mask=(frequency>=lo)&(frequency<=hi)
                        a,b=reference[...,mask].flatten(1),paired[...,mask].flatten(1)
                        corr=(a.conj()*b).sum(-1).abs() / (
                            torch.linalg.vector_norm(a, dim=-1)*torch.linalg.vector_norm(b, dim=-1)).clamp_min(1e-12)
                        sums[name][label+mode][k] += corr.double().sum()
        n_rows+=len(x)
assert n_rows==500 and len(SOURCE)==5
print(json.dumps(dict(source_U_holdout=n_rows, band_khz=BANDS, seed=2026112415,
                      quantity='Mean physical-packet magnitude of complex coherence of reference and observed companion',
                      target_reads=0, source_V_reads=0,
                      results={k:{label:(val/n_rows).tolist() for label,val in fields.items()}
                               for k,fields in sums.items()})))
