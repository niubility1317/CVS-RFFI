"""Disposable source-free CE/gradient/runtime verification; no checkpoint/IQ."""
import argparse,json,sys
from pathlib import Path
import torch
from torch.nn import functional as F
from experiments.cvs_coupled_identity.model import build,VARIANTS
from experiments.cvs_coupled_identity.physics import frozen_synthetic_diagnostics
from experiments.cvs_equivariant_identity.precision import numerical_context,FULL_FP32_POLICY


def run():
    torch.set_num_threads(2);rows=[]
    with numerical_context(FULL_FP32_POLICY):
        for seed in range(2026092701,2026092705):
            for variant in VARIANTS:
                torch.manual_seed(seed);model=build(variant).cpu()
                opt=torch.optim.AdamW(model.parameters(),lr=.0002,weight_decay=.0001);losses=[]
                for kind in ('random','weak','constant'):
                    x=torch.randn(4,2,256)
                    if kind=='weak':x*=1e-9;x[0].zero_()
                    if kind=='constant':x.fill_(.2)
                    x.requires_grad_(True);opt.zero_grad(set_to_none=True)
                    loss=F.cross_entropy(model(x),torch.arange(4));loss.backward()
                    assert torch.isfinite(loss) and torch.isfinite(x.grad).all()
                    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
                    opt.step();losses.append(dict(kind=kind,ce=float(loss.detach())))
                model.eval();d=frozen_synthetic_diagnostics(model)
                assert len(d['records'])==30 and not d['model_updated']
                rows.append(dict(seed=seed,variant=variant,losses=losses,contract=model.contract(),frozen_synthetic=d))
    return dict(status='PASS',python=sys.executable,torch_version=torch.__version__,device='cpu',
        disposable_models=8,disposable_ce_updates=24,formal_data_access=False,target_access=False,checkpoint_access=False,rows=rows)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    result=run();a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='rows'}))
