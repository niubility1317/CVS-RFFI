"""Focused physical and architecture checks; no real source or target data."""
import argparse
from copy import deepcopy
import io
import json
import torch
from experiments.cvs_dual_evidence import design as d
from experiments.cvs_dual_evidence.model import curvature_view,upgrade
from experiments.cvs_dual_evidence.runtime import installed
from experiments.cvs_phase1_stack.runtime import installed as legacy_installed


def checks(device='cpu'):
    from experiments.cvs_equivariant_identity.precision import numerical_context,actual_flags
    with numerical_context(d.FULL_FP32_POLICY):
        result=_checks(device);result['backend_flags']=actual_flags();return result


def _checks(device):
    torch.set_num_threads(2);g=torch.Generator().manual_seed(903)
    x=torch.randn(8,2,256,generator=g).to(device);z=torch.complex(x[:,0],x[:,1])
    t=torch.arange(256,device=device);changed=2.3*z*torch.exp(1j*(.7+.032*t))[None,:]
    transformed=torch.stack([changed.real,changed.imag],1)
    error=float((curvature_view(x)-curvature_view(transformed)).abs().max())
    assert error<2e-5,error
    zero=torch.zeros(2,2,256,device=device,requires_grad=True);curvature_view(zero).sum().backward()
    assert torch.isfinite(zero.grad).all()
    findings=[]
    c=d.config(d.rows()[0]);a=d.make_args(c,device);a.num_domains=15;a.num_classes=6;a.input_len=256
    with legacy_installed(c) as n:
        torch.manual_seed(c['model_seed']);original=n.build_baseline_model(a,torch.device(device)).eval()
    original_state={k:v.clone() for k,v in original.id_backbone.state_dict().items()}
    with torch.no_grad():reference=original(x)
    for arm in d.ARMS:
        model=deepcopy(original);cpu=torch.random.get_rng_state().clone()
        cuda=[v.clone() for v in torch.cuda.get_rng_state_all()] if torch.cuda.is_available() else []
        model=upgrade(model,arm,c['model_seed']).eval()
        assert torch.equal(cpu,torch.random.get_rng_state())
        assert all(torch.equal(v,w) for v,w in zip(cuda,torch.cuda.get_rng_state_all()))
        assert all(torch.equal(v,model.id_backbone.state_dict()[k]) for k,v in original_state.items())
        with torch.no_grad():
            result=model(x);assert torch.allclose(result,reference,atol=1e-6,rtol=1e-6)
        if arm!='legacy_cosine':
            model.train();opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.001)
            for _ in range(3):
                opt.zero_grad();out=model(x,return_aux=True)
                torch.nn.functional.cross_entropy(out['tx_logits'],torch.arange(len(x),device=device)%6).backward()
                assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
                opt.step()
            assert sum(float(p.grad.abs().sum()) for p in model.dom_backbone.parameters() if p.grad is not None)>0
            assert sum(float(p.grad.abs().sum()) for p in model.id_backbone.parameters() if p.grad is not None)>0
        model.eval();snapshot={k:v.clone() for k,v in model.state_dict().items()}
        with torch.no_grad():
            together=model(x);separate=torch.cat([model(row[None]) for row in x])
            assert torch.allclose(together,separate,atol=2e-5,rtol=2e-5), dict(arm=arm,max_batch_difference=float((together-separate).abs().max()),cudnn_tf32=torch.backends.cudnn.allow_tf32)
            assert all(torch.equal(v,model.state_dict()[k]) for k,v in snapshot.items())
            ema=deepcopy(model);assert torch.equal(ema(x),together)
            b=io.BytesIO();torch.save(model.state_dict(),b);b.seek(0);ema.load_state_dict(torch.load(b,map_location=device,weights_only=False),strict=True)
            assert torch.equal(ema(x),together)
        findings.append(dict(arm=arm,parameters=sum(p.numel() for p in model.parameters()),
            trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),status='PASS'))
    return dict(status='PASS',device=device,curvature_gain_phase_CFO_max_error=error,rows=findings,
        target_read=False,zero_gradient_finite=True,private_RNG=True,initial_main_logit_equivalence=True,
        both_backbone_gradients=True,stateless_batch_independent_inference=True,EMA_and_checkpoint_roundtrip=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--device',default='cpu');p.add_argument('--output',required=True);args=p.parse_args()
    result=checks(args.device);d.write(args.output,result);print(json.dumps(result))
