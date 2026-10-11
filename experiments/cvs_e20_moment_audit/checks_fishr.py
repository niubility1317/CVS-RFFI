"""CPU finite-population Monte Carlo and virtual-gradient identity checks."""
import argparse
import json
from pathlib import Path
import torch
from .fishr_stats import fishr_audit,unbiased_variances,exact_variances,penalty,sample_gradient


def run():
    torch.set_num_threads(2);generator=torch.Generator().manual_seed(1881)
    K,N,m,P,R=6,20,2,3,20000
    pool=torch.randn(2,K,N,P,generator=generator,dtype=torch.float64)
    pool=pool+torch.arange(K,dtype=torch.float64)[None,:,None,None]*.6
    pool[1]*=1.2
    means=pool.mean(2);within=pool.var(2,unbiased=False);between=(means-means.mean(1,keepdim=True)).square().mean(1)
    truth=within.mean(1)+between;exact=float(penalty(truth))
    def draw():
        order=torch.rand(R,2,K,N,generator=generator,dtype=torch.float64).argsort(-1)[...,:m]
        sample=pool[None].expand(R,-1,-1,-1,-1).gather(3,order[...,None].expand(-1,-1,-1,-1,P))
        pooled=sample.flatten(2,3)
        biased=pooled.var(2,unbiased=False);bessel=pooled.var(2,unbiased=True)
        mu=sample.mean(3);corrected=(mu-mu.mean(2,keepdim=True)).square().mean(2)+(1-(K-1)/(K*m)-1/(K*N))*sample.var(3,unbiased=True).mean(2)
        return biased,bessel,corrected
    old,bessel,va=draw();_,_,vb=draw()
    var_mean=(1-m/N)/(K*m)*pool.var(2,unbiased=True).mean(1)
    predicted_old=truth-var_mean;predicted_bessel=predicted_old*(K*m)/(K*m-1)
    for samples,expected in [(old,predicted_old),(bessel,predicted_bessel),(va,truth)]:
        se=samples.std(0)/R**.5
        assert bool(((samples.mean(0)-expected).abs()<=6*se+1e-10).all())
    centered_a=va-va.mean(1,keepdim=True);centered_b=vb-vb.mean(1,keepdim=True)
    cross=(centered_a*centered_b).mean((1,2));corrected_square=centered_a.square().mean((1,2))
    assert abs(float(cross.mean())-exact)<6*float(cross.std())/R**.5
    assert float(corrected_square.mean())>exact and bool((cross<0).any())
    # Scaling each parameter-gradient coordinate by t makes variance t^2 and
    # cross penalty t^4. Both batch paths must contribute the full derivative.
    t=torch.tensor(1.3,dtype=torch.float64,requires_grad=True)
    loss=((centered_a*t.square())*(centered_b*t.square())).mean()
    derivative=torch.autograd.grad(loss,t)[0]
    torch.testing.assert_close(derivative,4*t.detach().pow(3)*cross.mean(),atol=1e-12,rtol=1e-12)
    expected_gradient=4*float(t.detach())**3*exact
    assert abs(float(derivative)-expected_gradient)<6*4*float(t.detach())**3*float(cross.std())/R**.5
    # Finite N boundary m=N reproduces the exact full-population moments.
    labels=torch.arange(K).repeat_interleave(N).repeat(2);rx=torch.tensor([1,3]).repeat_interleave(K*N)
    g=pool.reshape(2*K*N,P);sizes={(r,k):N for r in (1,3) for k in range(K)}
    torch.testing.assert_close(unbiased_variances(g,labels,rx,sizes),exact_variances(g,labels,rx),atol=1e-12,rtol=1e-12)
    for bad_g,bad_y,bad_rx,bad_sizes in [(g[::N],labels[::N],rx[::N],sizes),(g,labels,rx,{key:1 for key in sizes})]:
        try:unbiased_variances(bad_g,bad_y,bad_rx,bad_sizes)
        except ValueError:pass
        else:raise AssertionError('Invalid m boundary accepted')
    # Check actual CE-head sample-gradient derivative through a feature affine.
    x=torch.randn(2*K*16,3,generator=generator,dtype=torch.float64)
    yy=torch.arange(K).repeat_interleave(16).repeat(2);rr=torch.tensor([1,3]).repeat_interleave(K*16)
    dd=torch.zeros(len(x),dtype=torch.long)
    with torch.random.fork_rng():
        torch.manual_seed(455);head=torch.nn.Linear(3,6).double()
    aa=torch.zeros(3,dtype=torch.float64,requires_grad=True);bb=torch.zeros(3,dtype=torch.float64,requires_grad=True)
    def exact_loss(a,b):return penalty(exact_variances(sample_gradient(x,head.weight,head.bias,yy,a,b),yy,rr))
    value=exact_loss(aa,bb);ga,gb=torch.autograd.grad(value,(aa,bb),retain_graph=True)
    assert all(v is None for v in torch.autograd.grad(value,tuple(head.parameters()),allow_unused=True))
    eps=1e-5;numeric=[]
    for coordinate in range(6):
        delta=torch.zeros(6,dtype=torch.float64);delta[coordinate]=eps
        numeric.append(float((exact_loss(delta[:3],delta[3:])-exact_loss(-delta[:3],-delta[3:]))/(2*eps)))
    torch.testing.assert_close(torch.cat((ga,gb)),torch.tensor(numeric,dtype=torch.float64),atol=1e-10,rtol=1e-6)
    state={k:v.detach().clone() for k,v in head.state_dict().items()};initial=x.clone();rng=torch.random.get_rng_state().clone()
    first=fishr_audit(x,head,yy,rr,dd,list(range(len(x))),99)
    second=fishr_audit(x,head,yy,rr,dd,list(range(len(x))),99)
    assert first==second and len(first['rounds'])==72 and torch.equal(rng,torch.random.get_rng_state())
    assert torch.equal(initial,x) and all(torch.equal(state[k],v) for k,v in head.state_dict().items())
    assert all(p.grad is None for p in head.parameters())
    assert any(v['cross_batch_PID_overlap']>0 for v in first['rounds'])
    for row in first['rounds']:
        if row['m']==16:
            assert row['cross_batch_PID_overlap']==len(x)
            for result in row['losses'].values():
                assert abs(result['error_vs_exact'])<1e-12 and result['gradient_squared_error_vs_exact']<1e-20
    try:fishr_audit(x[:10],head,yy[:10],rr[:10],dd[:10],list(range(10)),99)
    except ValueError:pass
    else:raise AssertionError('Incomplete population accepted')
    json.dumps(first,allow_nan=False)
    return dict(status='PASS',device='CPU',source_data_used=False,target_access=False,monte_carlo_replicates=R,
        exact_finite_population_penalty=exact,pooled_n_mean_bias=float((old.mean(0)-truth).mean()),
        pooled_n_minus_1_mean_bias=float((bessel.mean(0)-truth).mean()),
        stratified_corrected_mean_bias=float((va.mean(0)-truth).mean()),
        corrected_square_loss_mean=float(corrected_square.mean()),cross_loss_mean=float(cross.mean()),cross_loss_standard_error=float(cross.std())/R**.5,
        cross_negative_fraction=float((cross<0).double().mean()),affine_scale_gradient=float(derivative),exact_expected_scale_gradient=expected_gradient,
        finite_population_m_equals_N_exact=True,both_batch_gradient_identity=True,real_CE_affine_gradient_finite_difference=True,
        real_head_gradient_absent=True,allowed_cross_batch_PID_overlap=True,seed_reproducible=True,input_head_and_RNG_unchanged=True,
        native_diagnostic_schema_rounds=len(first['rounds']),no_backbone_gradient_claim=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args();result=run()
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps(result,allow_nan=False))
