"""Data-free paired initialization and CE gradient smoke, never RF accuracy."""
import argparse,copy,json
from pathlib import Path
import torch
from torch.nn import functional as F
from experiments.cvs_equivariant_identity.model import rotate_pair
from experiments.cvs_neural_residual_identity.model import build as control
from experiments.cvs_frontfilter_identity.model import VARIANTS,build,filter_contract

SEEDS=(2026092701,2026092702,2026092703,2026092704)

def run(output):
    output=Path(output)
    if output.exists():raise FileExistsError('Preserve smoke evidence')
    torch.set_num_threads(2)
    x=torch.randn(4,2,256,generator=torch.Generator().manual_seed(2026100300))
    x[0].zero_();x[1].fill_(1.)
    t=torch.arange(256,dtype=x.dtype)
    x[2]=torch.stack([torch.cos(.13*t)+.15*torch.cos(.39*t),torch.sin(.13*t)+.15*torch.sin(.39*t)])
    labels=torch.arange(4);records=[];bases={}
    for variant in VARIANTS:
        for seed in SEEDS:
            torch.manual_seed(seed);base=control('neural_residual_shallow').eval();rng=torch.get_rng_state().clone()
            torch.manual_seed(seed);model=build(variant).eval()
            assert torch.equal(rng,torch.get_rng_state())
            assert all(torch.equal(v,model.state_dict()[k]) for k,v in base.state_dict().items())
            if variant==VARIANTS[0]:bases[seed]=model.frontfilter.basis.detach().clone()
            else:assert torch.equal(bases[seed],model.frontfilter.basis)
            with torch.no_grad():assert torch.equal(model(x),base(x))
            assert model.contract()==filter_contract(variant)
            optimizer=torch.optim.AdamW(model.parameters(),lr=.0002,weight_decay=.0001)
            model.train();steps=[]
            for step in range(3):
                optimizer.zero_grad(set_to_none=True);loss=F.cross_entropy(model(x),labels);loss.backward()
                assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
                gradients={n:float(p.grad.norm()) for n,p in model.frontfilter.named_parameters()}
                assert any(v>0 for v in gradients.values())
                if step==2:assert all(v>0 for v in gradients.values())
                optimizer.step()
                steps.append(dict(step=step+1,cross_entropy=float(loss.detach()),frontfilter_gradient_norms=gradients))
            model.eval();state={k:v.clone() for k,v in model.state_dict().items()};rng=torch.get_rng_state().clone()
            diagnostics=model.diagnostics(x)
            assert all(torch.equal(v,model.state_dict()[k]) for k,v in state.items())
            assert torch.equal(rng,torch.get_rng_state())
            assert diagnostics['frontfilter']['records'][0]['relative_input_change_mean']>0
            with torch.no_grad():
                logits=model(x);single=torch.cat([model(row[None]) for row in x])
                phase=model(rotate_pair(x,torch.tensor([.43,-.6,1.2,-2.])))
            torch.testing.assert_close(logits,single,atol=2e-4,rtol=2e-4)
            # Pure DC + zero-padded near-identity FIR creates tiny edge spectra.
            # The retained spectral log-ratio path amplifies FP32 roundoff.
            # Preserve this measured limitation rather than relax its tolerance.
            ordinary=torch.tensor([True,False,True,True])
            torch.testing.assert_close(logits[ordinary],phase[ordinary],atol=2e-3,rtol=2e-3)
            assert torch.isfinite(logits).all() and torch.isfinite(phase).all()
            model64=copy.deepcopy(model).double()
            with torch.no_grad():
                dc=x[1:2].double();theta=dc.new_tensor([-.6])
                dc64=model64(dc);phase64=model64(rotate_pair(dc,theta))
            torch.testing.assert_close(dc64,phase64,atol=2e-5,rtol=2e-5)
            records.append(dict(variant=variant,model_seed=seed,status='VERIFIED',parameters=sum(p.numel() for p in model.parameters()),
                new_trainable_parameters=sum(p.numel() for p in model.frontfilter_parameters()),
                initial_function_exact=True,old_state_exact=True,cpu_rng_preserved=True,paired_basis_initialization=True,
                contract=model.contract(),steps=steps,diagnostics=diagnostics,inference_state_unchanged=True,
                common_phase_max_abs_error=float((logits-phase).abs().max()),
                ordinary_fp32_phase_max_abs_error=float((logits[ordinary]-phase[ordinary]).abs().max()),
                constant_fp32_phase_max_abs_error=float((logits[1]-phase[1]).abs().max()),
                constant_same_state_fp64_phase_max_abs_error=float((dc64-phase64).abs().max()),
                constant_fp32_uniform_tolerance_claim=False,
                numerical_scope='Ordinary noise/tone/zero FP32 checked; pureDC FP32 reported with same-state FP64 confirmation. Retained spectral log-ratio sensitivity is not repaired or hidden.',
                packet_independence_max_abs_error=float((logits-single).abs().max())))
    result=dict(status='VERIFIED',data_access=False,target_access=False,checkpoint_loading=False,public_synthetic_input=True,loss='single cross_entropy',records=records)
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x',encoding='utf-8',newline='\n') as stream:
        json.dump(result,stream,ensure_ascii=False,indent=2,allow_nan=False);stream.write('\n')
    assert len(json.loads(output.read_text(encoding='utf-8'))['records'])==8
    print(output);return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);run(p.parse_args().output)
