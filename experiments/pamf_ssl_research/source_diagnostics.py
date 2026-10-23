"""Frozen source V diagnostics; no teacher, prototype or model-state update."""
from pathlib import Path
import numpy as np
import torch
from experiments.cric_dura_comparison.io import read, write


def run(c, output, device='cuda:0'):
    from experiments.cric_dura_comparison.frozen_inference import load_F
    from experiments.cric_dura_comparison.f_data import SourceDataset
    from experiments.cric_dura_comparison.t1_practical import Practical
    contract = read(c['source_contract'])
    data = SourceDataset(c['source_cache'], 'V', contract)
    chosen = []
    m = data.metadata
    for cls in range(6):
        for rx in c['source_rxs']:
            for day in c['source_days']:
                index = np.flatnonzero((m['y'] == cls) & (m['rx_i'] == rx) & (m['day_i'] == day))
                chosen.extend(index[:8].tolist())
    if len(chosen) != 720:
        raise ValueError('Fixed source-V diagnostic coverage differs')
    reader, _ = load_F(c['output_root'], contract, device)
    simulator = Practical(c)
    truth, clean, channel = [], [], []
    with torch.inference_mode():
        for begin in range(0, len(chosen), 64):
            ids = chosen[begin:begin + 64]
            x = data.x[ids].to(device)
            meta = {k: torch.tensor(m[k][ids].tolist(), device=device) for k in ('rx_i', 'day_i', 'eq_i', 'sig_i', 'base_index')}
            strong = simulator(x, meta, 50, 900000 + begin)
            truth.extend(m['y'][ids].tolist())
            for samples, result in ((x, clean), (strong, channel)):
                result.extend(reader.classify_features(reader.features(samples)).argmax(1).cpu().tolist())
    y = np.asarray(truth)
    write(output, dict(status='FROZEN_SOURCE_V_DIAGNOSTIC', samples=720,
        physical_ids=m['ids'][chosen].tolist(), clean_accuracy=float((np.asarray(clean) == y).mean()),
        fixed_practical_mixture_accuracy=float((np.asarray(channel) == y).mean()),
        model_state_updates=0, prototype_updates=0, teacher_updates=0,
        target_reads=0, U_truth_reads=0, head_fit_updates=0,
        source_output=c['output_root'], simulator_scene_coverage=dict(simulator.coverage)))
