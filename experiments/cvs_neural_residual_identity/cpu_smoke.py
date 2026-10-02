"""Data-free smoke for exact baseline initialization and learnable residuals."""
import argparse,json
from pathlib import Path
import torch
from torch.nn import functional as F
from experiments.cvs_neural_residual_identity.model import build,VARIANTS,neural_contract
from experiments.cvs_adaptive_volterra_identity.model import build as control
from experiments.cvs_neural_residual_identity.physics import frozen_synthetic_diagnostics

def run(output):
    torch.set_num_threads(2);records=[]
    for variant in VARIANTS:
        torch.manual_seed(2026092701);base=control('adaptive_volterra_lag4').eval()
        torch.manual_seed(2026092701);m=build(variant).eval()
        x=torch.randn(4,2,256);x[0].zero_();x[1].fill_(1);y=torch.tensor([0,1,2,3])
        assert m.contract()==neural_contract(variant)
        assert torch.equal(m(x),base(x))
        opt=torch.optim.AdamW(m.parameters(),lr=.0002,weight_decay=.0001)
        for step in range(2):
            opt.zero_grad(set_to_none=True);loss=F.cross_entropy(m(x),y);loss.backward()
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())
            if step==1:assert all(b.expand.weight_real.grad.norm()>0 for _,b in m.residual_blocks())
            opt.step()
        physics=frozen_synthetic_diagnostics(m)
        assert physics['target_access'] is False and physics['training_augmentation'] is False
        assert len(physics['neural_residual']['records'])==2*neural_contract(variant)['neural_depth_per_path']
        records.append(dict(variant=variant,parameters=sum(p.numel() for p in m.parameters()),ce=float(loss.detach()),
            initial_exact=True,second_step_hidden_CE_gradient=True,phase_tolerance_pass=physics['phase_tolerance_pass'],checkpoint_source=None))
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(dict(status='VERIFIED',data_access=False,records=records),indent=2)+'\n',encoding='utf-8')
    print(output)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);run(p.parse_args().output)
