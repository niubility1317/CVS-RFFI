"""Real scratch state save/reload, forward/backward; no source or target data."""
import argparse,json,tempfile
from pathlib import Path
import torch
from experiments.cvs_validdual_identity.model import build,VARIANTS,dual_contract
from experiments.cvs_validdual_identity.source import validate_config
from experiments.cvs_validdual_identity.physics import frozen_synthetic_diagnostics
ROOT=Path(__file__).resolve().parents[2]
def smoke(output):
    if output.exists():raise FileExistsError('Preserve smoke evidence')
    torch.set_num_threads(2);rows=[]
    for variant in VARIANTS:
        c=json.loads((ROOT/'experiments/cvs_validdual_identity/configs'/f'{variant}-s2026092701.json').read_text(encoding='utf-8'));validate_config(c)
        torch.manual_seed(13);m=build(variant);x=torch.randn(3,2,256)
        assert m.contract()==dual_contract(variant)
        loss=torch.nn.functional.cross_entropy(m(x),torch.arange(3));loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters() if p.requires_grad)
        with tempfile.TemporaryDirectory(prefix='cvs_validdual_') as d:
            path=Path(d)/'scratch.pt';torch.save(m.state_dict(),path);clone=build(variant)
            clone.load_state_dict(torch.load(path,map_location='cpu',weights_only=False),strict=True)
            m.eval();clone.eval();assert torch.equal(m(x),clone(x))
        physics=frozen_synthetic_diagnostics(m)
        rows.append(dict(variant=variant,parameters=sum(p.numel() for p in m.parameters()),gradient_used_parameters=sum(p.numel() for p in m.parameters() if p.grad is not None),
            loss=float(loss.detach()),finite=True,checkpoint_reload=True,physics=physics))
    with output.open('x',encoding='utf-8') as f:json.dump(dict(status='VERIFIED',rows=rows,source_access=False,target_access=False),f,indent=2)
    print(json.dumps(dict(status='VERIFIED',variants=list(VARIANTS))))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();smoke(a.output)
