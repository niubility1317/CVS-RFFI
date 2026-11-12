"""CPU-only strict load and synthetic forward for frozen Tweak/ISSL sources."""
import json
from pathlib import Path
import sys

ROOT=Path('/home/szu2070436088/2510044040/CV-SincNet')
sys.path.insert(0,str(ROOT/'releases/20261109-phase2-bsource-savedb-mopc-m4-r15-release'))
import torch
from experiments.pamf_phase2_practical import external_additional as extra


def main():
    torch.set_num_threads(1);torch.manual_seed(20261112)
    runtime=json.loads((ROOT/'releases/20261029-phase2-external-threepool-m4-r01-release/runtime.json').read_text())
    items=[]
    for profile in ('tweak_m1','issl_fixed_v1'):
        for seed in (2026092701,2026092702,2026092703,2026092704):
            row=dict(profile=profile,seed=seed)
            try:
                state=extra.load_source(runtime,profile,seed,'cpu')
                model=state['model'].eval();before={k:v.clone() for k,v in model.state_dict().items()}
                with torch.inference_mode():
                    z=state['score_fn'](model,torch.randn(1,2,256))
                if z.shape!=(1,6) or not bool(torch.isfinite(z).all()):raise ValueError('bad synthetic output')
                if any(not torch.equal(before[k],v) for k,v in model.state_dict().items()):raise ValueError('model changed')
                row.update(status='PASS',classes=6,source_path=state['source_path'],synthetic_rows=1)
            except Exception as exc:row.update(status='FAILED',error=type(exc).__name__+': '+str(exc))
            items.append(row)
    print(json.dumps(dict(status='PASS' if all(x['status']=='PASS' for x in items) else 'PARTIAL',items=items,
                          actual_target_rows=0,actual_query_rows=0,truth_read=False,training=False)))


if __name__=='__main__':main()
