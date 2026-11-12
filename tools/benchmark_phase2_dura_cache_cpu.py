"""Synthetic CPU eager benchmark of shared DURA restoration, not full-run timing."""
import json
from pathlib import Path
import sys
import time

ROOT=Path('/home/szu2070436088/2510044040/CV-SincNet')
RELEASE=ROOT/'releases/20261101-phase2-r42-group1-m4-r04-release'
sys.path.insert(0,str(RELEASE));sys.path.insert(0,str(RELEASE/'code'))
import torch
from experiments.pamf_phase2_practical.io import activate,read


def main():
    torch.set_num_threads(1);torch.manual_seed(20261112);rt=read(RELEASE/'runtime.json');activate(rt)
    from experiments.cric_dura_comparison.frozen_inference import load_G
    from experiments.pamf_stage2_inference.engine import InferenceEngine
    model=load_G(rt['DURA'],read(rt['source_contract']),'cpu');engine=InferenceEngine(model,reuse=True,graph=False,native=False)
    x=torch.randn(8,2,256);rounds=[]
    with torch.inference_mode():
        engine(x)
        for _ in range(3):
            start=time.perf_counter();reference=[engine(x)[0] for _ in range(3)];baseline=time.perf_counter()-start
            start=time.perf_counter();cached=engine(x)[0];values=[cached.clone() for _ in range(3)];cached_s=time.perf_counter()-start
            if not all(torch.equal(a,b) for a,b in zip(reference,values)):raise ValueError('Cache output differs')
            rounds.append(dict(recomputed_seconds=baseline,cached_seconds=cached_s,
                               recomputed_G_calls=3,cached_G_calls=1))
    print(json.dumps(dict(status='PASS',scope='synthetic_8_records_CPU_eager_component_only',rounds=rounds,
          actual_target_rows=0,actual_query_rows=0,truth_read=False,training=False,
          performance_claim='not GPU or end-to-end throughput')))


if __name__=='__main__':main()
