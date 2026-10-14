"""Actual-constructor pinned-readout test on synthetic IQ, without checkpoint."""
import copy
import json
import torch
from . import design as d
from .source_probe import evaluate
from experiments.cvs_evidence_recomposition20.capture import forward
from experiments.cvs_evidence_recomposition20.source_probe import protected_names
from experiments.cvs_u_schedule50.counterfactual import _equal,_rng_snapshot


def main():
    from experiments.cvs_crossrx_sharpness20 import model as api
    torch.set_num_threads(2)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(2026101519)
        c=d.parent.config(next(r for r in d.parent.rows() if r['row_id']=='BASE_CE-s2026092701'))
        model=api.build(c,torch.device('cpu')).eval()
        named=[(n,p) for n,p in model.named_parameters() if p.requires_grad]
        names=[n for n,p in named];pre=tuple(p.detach().clone() for n,p in named);protected=protected_names(model)
        cells=[(y,rx,day) for side in (0,1) for y in range(6) for rx in d.SOURCE_RX for day in (1,2,3)]
        bank=dict(x=torch.randn(180,2,256),y=torch.tensor([v[0] for v in cells]),
            rx=torch.tensor([v[1] for v in cells]),day=torch.tensor([v[2] for v in cells]))
        before=copy.deepcopy(model.state_dict());rng=_rng_snapshot()
        with torch.no_grad():actual=forward(model,bank['x'])['joint']
        z,b=evaluate(model,names,pre,bank,protected)
        if not torch.equal(z,actual):raise AssertionError('Evaluation did not use the actual deployed joint')
        poisoned=list(pre)
        for i,name in enumerate(names):
            if name in protected:poisoned[i]=pre[i]+.5
        same,q=evaluate(model,names,tuple(poisoned),bank,protected)
        if not torch.equal(same,z) or q!=b:raise AssertionError('Updated head/gains leaked into pinned feature evaluation')
        head=model.id_backbone.encoder.core.id_backbone.cls_head
        index=next(i for i,(n,p) in enumerate(named) if p is head.base_norm.weight)
        changed=list(pre);changed[index]=pre[index].clone();changed[index][0]+=.1
        new,e=evaluate(model,names,tuple(changed),bank,protected)
        rms=float((new.double()-z.double()).square().mean().sqrt())
        if rms<=1e-7:raise AssertionError('Actual feature parameter change was hidden by readout evaluation')
        if (not _equal(before,model.state_dict()) or not _equal(rng,_rng_snapshot())
            or any(p.grad is not None for p in model.parameters()) or any(m.training for m in model.modules())):
            raise AssertionError('Evaluation changed live model/gradient/flags/RNG')
    print(json.dumps(dict(status='PASS',actual_constructor_joint_bitwise=True,
        updated_W_all_gains_ignored_in_fixed_readout=True,actual_feature_change_RMS=rms,
        parameters_buffers_gradients_flags_RNG_preserved=True,balanced_raw_geometry_legal=True,
        synthetic_packet_forwards=720,data_checkpoint_remote_reads=0,head_fit_updates=0)),flush=True)


if __name__=='__main__':main()
