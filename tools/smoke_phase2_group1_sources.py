"""Strict CPU source-checkpoint loading and synthetic forward, no target data."""
import json
from pathlib import Path
import sys

ROOT=Path('/home/szu2070436088/2510044040/CV-SincNet')
sys.path.insert(0,str(ROOT/'releases/20261109-phase2-bsource-savedb-mopc-m4-r15-release'))
import torch
from experiments.pamf_phase2_practical import group1_source


def main():
    torch.set_num_threads(1);torch.manual_seed(20261112)
    contract=str(ROOT/'runs/20261021-phase1-CRIC-DURA-comparison-m153-r02/data/F0/source_contract.json')
    profiles=('protonet_cda','mrior_sda','dadda_sda','twostage','poster_ft','mopc_hr','lora_triplet_knn')
    items=[]
    for profile in profiles:
        source='20261101-phase2-mopc-author-group1-m4-r04' if profile=='mopc_hr' else '20261030-phase2-group1-concat-midPL-threepool-m4-r01'
        for seed in (2026092701,2026092702,2026092703,2026092704):
            row=dict(profile=profile,seed=seed)
            try:
                cache=str(ROOT/'runs/20261023-phase1-practical-residual-full2x2-m600-r01/data/source/F0')
                bundle=group1_source.load_source(dict(run_root=str(ROOT/'runs'/source),source_contract=contract,
                                                       source_cache=cache),profile,seed,'cpu')
                model=bundle['model'].eval();before={k:v.clone() for k,v in model.state_dict().items()}
                with torch.inference_mode():z=bundle['score_fn'](model,torch.randn(1,2,256))
                if z.shape!=(1,6) or not bool(torch.isfinite(z).all()):raise ValueError('Bad synthetic forward')
                if any(not torch.equal(before[k],v) for k,v in model.state_dict().items()):raise ValueError('Model changed')
                row.update(status='PASS',source_path=bundle.get('source_checkpoint'),synthetic_rows=1)
            except Exception as exc:row.update(status='FAILED',error=type(exc).__name__+': '+str(exc))
            items.append(row)
    print(json.dumps(dict(status='PASS' if all(x['status']=='PASS' for x in items) else 'PARTIAL',items=items,
          actual_target_rows=0,actual_query_rows=0,truth_read=False,training=False)))


if __name__=='__main__':main()
