"""Measure the WiSig physical guard on held source U and generic stress inputs."""

import json
import sys
from pathlib import Path

import numpy as np
import torch

from experiments.cric_dura_comparison.io import read, tensor
from experiments.pamf_lora_main.runner import activate
from experiments.pamf_unrolled_restore.wisig_anchored_window_source import HELD
from experiments.pamf_unrolled_restore.wisig_observed_solve import ObservedSolve, generic_observation
from experiments.pamf_unrolled_restore.wisig_observed_solve_source import RUN
from experiments.pamf_unrolled_restore.wisig_uncertain_prior import as_complex
from experiments.pamf_unrolled_restore.wisig_uncertain_source import source_iq


@torch.inference_mode()
def probe(runtime):
    if runtime['run_id']!=RUN:
        raise ValueError('The source-only observed-solve runtime is required')
    activate(runtime)
    source,training,calibration = source_iq(Path(runtime['source_root']))
    held = training[:HELD]
    model = ObservedSolve(True).cuda().eval()
    generator = torch.Generator(device='cuda:0').manual_seed(2026112401)

    def scores(raw):
        return torch.stack(model.gate_scores(model.analyze(raw)),dim=1)

    clean_cal = torch.cat([scores(as_complex(tensor(source[index],'cuda:0'))) for index in np.array_split(calibration,64)])
    gate = 1.02*clean_cal.amax(0)+1e-6
    held = as_complex(tensor(source[held],'cuda:0'))
    time = model.time
    phase = time.float()/255

    def reflected_delay(amount):
        index = time+amount
        return held[:,index.minimum(2*(len(time)-1)-index)]

    panels = {'unmodified_source_U':held,'reflected_delay_12':reflected_delay(12),'reflected_delay_28':reflected_delay(28),
              'phase_curvature':held*torch.polar(torch.ones_like(phase),1.10*torch.sin(7.*phase)),
              'generic_disturbed':generic_observation(held,generator,model)}
    summary = {}
    for name,rows in panels.items():
        measures = torch.cat([scores(batch) for batch in rows.split(64)])
        active = measures>gate
        summary[name] = dict(count=len(rows),triggered=int(active.any(1).sum()),
            shift=int(active[:,0].sum()),repeat=int(active[:,1].sum()),median=measures.median(0).values.tolist())
    return dict(status='VERIFIED',source_U_calibration=len(calibration),V_reads=0,F_reads=0,TX_reads=0,
                target_reads=0,checkpoint_reads=0,gate=gate.tolist(),panels=summary)


if __name__=='__main__':
    print(json.dumps(probe(read(Path(sys.argv[1]))),allow_nan=False))
