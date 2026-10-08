"""No-dataset numerical, gradient, branch and inference-contract checks."""
import argparse
import json
from pathlib import Path
import tempfile
import torch
import torch.nn.functional as F
from experiments.cvs_reference_identity.model import build
from experiments.cvs_equivariant_identity.precision import numerical_context
from experiments.cvs_receiver_residual import design as d
from experiments.cvs_receiver_residual.objectives import leave_tx_targets, auxiliary_losses


def smoke(device='cpu'):
    torch.set_num_threads(2); device=torch.device(device); rows=[]
    for arm in d.ARMS:
        torch.manual_seed(37); base=build('reference_response').to(device).eval()
        expected_rng=torch.random.get_rng_state().clone()
        expected_cuda_rng=torch.cuda.get_rng_state(device).clone() if device.type=='cuda' else None
        torch.manual_seed(37); model=d.build_model(arm).to(device).eval()
        if not torch.equal(expected_rng,torch.random.get_rng_state()):
            raise AssertionError('Added head initialization changed shared RNG')
        if expected_cuda_rng is not None and not torch.equal(expected_cuda_rng,torch.cuda.get_rng_state(device)):
            raise AssertionError('Added head initialization changed CUDA dropout RNG')
        for k,v in base.state_dict().items():
            if not torch.equal(v,model.encoder.state_dict()[k]):
                raise AssertionError('Backbone initialization differs '+k)
        x=torch.randn(60,2,256,device=device)
        model.strength.fill_(1.)
        with torch.no_grad():
            result=model.details(x)
            torch.testing.assert_close(base(x),result['logits'],rtol=0,atol=0)
            torch.testing.assert_close(result['branches'].sum(1),result['base'],rtol=1e-6,atol=1e-6)
            torch.testing.assert_close(model(x[:1]),model(x)[:1],rtol=2e-5,atol=2e-5)
            if not torch.isfinite(model(torch.zeros(2,2,256,device=device))).all():
                raise AssertionError('Zero IQ nonfinite')
        y=torch.arange(6,device=device).repeat_interleave(10)
        rx=torch.tensor(d.SOURCE_RXS,device=device).repeat(12)
        centroids=torch.randn(6,5,160,device=device)
        targets=leave_tx_targets(centroids)
        modified=centroids.clone();modified[2]+=1000
        torch.testing.assert_close(targets[2],leave_tx_targets(modified)[2],rtol=0,atol=0)
        bank=dict(targets=targets.cpu(),scale=torch.tensor(1.))
        model.train();details=model.details(x)
        aux,info=auxiliary_losses(model,details,y,rx,bank,1.)
        loss=F.cross_entropy(details['logits'],y)+aux;loss.backward()
        gradient={}
        for name in ('displacement','contribution'):
            if not hasattr(model,name): continue
            values=[p.grad.norm() for p in getattr(model,name).parameters() if p.grad is not None]
            norm=float(torch.stack(values).norm())
            if not norm>0: raise AssertionError('Missing auxiliary head gradient '+name)
            gradient[name]=norm
        if any(not torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):
            raise AssertionError('Nonfinite gradients')
        torch.optim.AdamW(model.parameters(),lr=.0002).step();model.eval()
        with torch.no_grad():
            after=model.details(x)
            torch.testing.assert_close(model(x[:1]),model(x)[:1],rtol=2e-5,atol=2e-5)
            if float(after['correction_norm_ratio'])>.250001 or float(after['gate_abs_delta'])>.250001:
                raise AssertionError('Residual cap violated')
            before=model(x[:2])
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'roundtrip.pt';torch.save(model.state_dict(),path)
            copy=d.build_model(arm).to(device).eval()
            copy.load_state_dict(torch.load(path,map_location=device,weights_only=False),strict=True)
            with torch.no_grad(): torch.testing.assert_close(copy(x[:2]),before,rtol=0,atol=0)
        rows.append(dict(arm=arm,baseline_bitwise_equivalence=True,checkpoint_roundtrip=True,
            samplewise_inference=True,gradient_norm=gradient,
            correction_norm_ratio=float(after['correction_norm_ratio']),gate_abs_delta=float(after['gate_abs_delta']),
            added_parameters=model.contract()['added_parameters'],cross_tx_pairs=info['cross_tx_pair_count']))
    # Exercise the exact existing concat sampler boundary with synthetic metadata.
    from experiments.cvs_phase1_overlay.augmentation import OriginalLEO
    with tempfile.TemporaryDirectory() as folder:
        augmentation=OriginalLEO(folder)
        batch=dict(meta=[dict(sample_id='synthetic-'+str(i),rx_i=int(rx[i]),day_i=1) for i in range(len(x))])
        for epoch in (1,41,80,91):
            view=augmentation(x,batch,epoch,1)
            if view.x.shape!=x.shape or not torch.isfinite(view.x).all():
                raise AssertionError('Native concat sampler boundary failed')
    bad=d.config('baseline',d.SEEDS[0]);bad['p1_truth']='forbidden'
    try: d.validate_config(bad)
    except ValueError: pass
    else: raise AssertionError('Source config accepted target truth')
    return dict(status='PASS',device=str(device),rows=rows,leave_tx_exclusion=True,
        target_input_rejection=True,native_concat_sampler=True,real_dataset_read=False,query_read=False)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--device',default='cpu');p.add_argument('--output',required=True)
    a=p.parse_args()
    with numerical_context(d.FULL_FP32_POLICY): result=smoke(a.device)
    d.write(a.output,result);print(json.dumps(result))
