"""Strict CPU PAMF/DURA checkpoint load and synthetic forward, no target data."""
import json
from pathlib import Path
import sys

ROOT=Path('/home/szu2070436088/2510044040/CV-SincNet')
RELEASE=ROOT/'releases/20261101-phase2-r42-group1-m4-r04-release'
sys.path.insert(0,str(RELEASE));sys.path.insert(0,str(RELEASE/'code'))
import torch
from experiments.pamf_phase2_practical.io import activate,read


def main():
    torch.set_num_threads(1);torch.manual_seed(20261112)
    rt=read(RELEASE/'runtime.json');activate(rt);items=[]
    from experiments.pamf_phase2_practical.trisig_source import load
    for seed in (2026092701,2026092702,2026092703,2026092704):
        row=dict(profile='PAMF',seed=seed)
        try:
            model=load(rt,seed,'cpu')['reader'].eval();before={k:v.clone() for k,v in model.state_dict().items()}
            with torch.inference_mode():x=torch.randn(1,2,256);z=model.classify_features(model.features(x))
            if z.shape!=(1,6) or not bool(torch.isfinite(z).all()):raise ValueError('bad synthetic forward')
            if any(not torch.equal(before[k],v) for k,v in model.state_dict().items()):raise ValueError('model changed')
            row.update(status='PASS',synthetic_rows=1)
        except Exception as exc:row.update(status='FAILED',error=type(exc).__name__+': '+str(exc))
        items.append(row)
    row=dict(profile='DURA_v1',seed=2026092701)
    try:
        from experiments.cric_dura_comparison.frozen_inference import load_G
        from experiments.pamf_stage2_inference.engine import InferenceEngine
        model=load_G(rt['DURA'],read(rt['source_contract']),'cpu')
        before={k:v.clone() for k,v in model.state_dict().items()}
        engine=InferenceEngine(model,reuse=True,graph=False,native=False)
        with torch.inference_mode():x=torch.randn(1,2,256);z=engine(x)[0]
        if z.shape!=x.shape or not bool(torch.isfinite(z).all()):raise ValueError('bad synthetic forward')
        if any(not torch.equal(before[k],v) for k,v in model.state_dict().items()):raise ValueError('model changed')
        row.update(status='PASS',synthetic_rows=1,runtime='CPU_eager_diagnostic_not_frozen_CUDA_graph')
    except Exception as exc:row.update(status='FAILED',error=type(exc).__name__+': '+str(exc))
    items.append(row)
    print(json.dumps(dict(status='PASS' if all(x['status']=='PASS' for x in items) else 'PARTIAL',items=items,
          actual_target_rows=0,actual_query_rows=0,truth_read=False,training=False)))


if __name__=='__main__':main()
