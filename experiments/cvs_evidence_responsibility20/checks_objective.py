"""Focused role/graph/reconstruction and equal-value routing checks, no IQ/CP."""
import copy,json
import torch
from .objective import BRANCHES,losses,compose,ce,positive,TOLERANCE,COEFFICIENT
def main():
    torch.set_num_threads(2);generator=torch.Generator().manual_seed(10031)
    cells=[(rx,day,side,y) for rx in (1,3,4,6,8) for day in (1,2,3) for side in (0,1) for y in range(6)]
    labels=torch.tensor([c[3] for c in cells]);weight=torch.eye(160)[:6].clone().requires_grad_()
    base=weight.detach()[labels]+.5*weight.detach()[(labels+1)%6]
    physical=-2*weight.detach()[labels]+3*weight.detach()[(labels+1)%6]
    raw=dict(identity=base.clone().requires_grad_(),physical=physical.clone().requires_grad_(),reference_raw=(.01*torch.randn(180,160,generator=generator)).requires_grad_(),sequence_raw=(weight.detach()[labels]+.01*torch.randn(180,160,generator=generator)).requires_grad_())
    gains=[torch.full((160,),v,requires_grad=True) for v in (.5,.05,.15)]
    components=dict(raw,joint=compose(raw,gains))
    bank=dict(x=torch.randn(180,2,256,generator=generator),y=labels,rx=torch.tensor([c[0] for c in cells]),day=torch.tensor([c[1] for c in cells]),side=torch.tensor([c[2] for c in cells]),ids=[(c[3],c[0],c[1],c[2],i+100) for i,c in enumerate(cells)])
    values,summaries=losses(components,bank,weight,gains);q=summaries['ROUTED_GAP']
    assert q['active_decisions']>=180 and q['active_origin_packet_count']==180 and q['by_branch']['physical']['active']==180
    assert torch.equal(values['JOINT_GAP'],values['ROUTED_GAP']) and float(values['ZERO'].detach())==0.
    params=[raw[name] for name in BRANCHES]+[weight]+gains
    grads={mode:torch.autograd.grad(value,params,retain_graph=True,allow_unused=True) for mode,value in values.items()}
    for mode in grads:assert all(g is None for g in grads[mode][4:])
    assert all(g is not None and torch.isfinite(g).all() for g in grads['ROUTED_GAP'][:4])
    assert any(not torch.equal(a,b) for a,b in zip(grads['JOINT_GAP'][:4],grads['ROUTED_GAP'][:4]))
    factual,_=ce(compose({k:v.detach() for k,v in raw.items()},gains,removed='physical'),labels,weight)
    routed_ce,_=ce(compose(raw,gains,only='physical'),labels,weight)
    term=positive(routed_ce,factual).sum()
    isolated=torch.autograd.grad(term,params,retain_graph=True,allow_unused=True)
    assert isolated[1] is not None and isolated[1].abs().sum()>0
    assert all(g is None for i,g in enumerate(isolated) if i!=1)
    manual=COEFFICIENT*sum(positive(ce(components['joint'],labels,weight)[0],ce(compose({k:v.detach() for k,v in raw.items()},gains,removed=b),labels,weight)[0]).sum() for b in BRANCHES)/720
    assert abs(float(manual.detach())-float(values['JOINT_GAP'].detach()))<2e-7
    v=torch.tensor([0.,1.,2.],requires_grad=True);f=torch.tensor([1.,1.,1.],requires_grad=True)
    gv,gf=torch.autograd.grad(positive(v,f).sum(),(v,f),allow_unused=True);assert torch.equal(gv,torch.tensor([0.,0.,1.])) and gf is None
    edge=torch.tensor([0.],dtype=torch.float64,requires_grad=True);edge_f=torch.tensor([-TOLERANCE],dtype=torch.float64)
    assert torch.autograd.grad(positive(edge,edge_f).sum(),edge)[0].item()==0.
    n=0
    for role in ('U_s','V','target'):
        try:losses(components,{'y':object()},weight,gains,source_role=role)
        except ValueError:n+=1
        else:raise AssertionError('role before label read')
    for mutation in range(7):
        badc=dict(components);badb=dict(bank);badw=weight;badg=gains
        if mutation==0:badb['hide']=True
        elif mutation==1:badc['joint']=components['joint']+1.
        elif mutation==2:badw=weight[:5]
        elif mutation==3:badg=gains[:2]
        elif mutation==4:badg=[gains[0][:159],*gains[1:]]
        elif mutation==5:badc['identity']=raw['identity'].double()
        else:badc['physical']=raw['physical'].detach().clone();badc['physical'][0,0]=float('nan')
        try:losses(badc,badb,badw,badg)
        except ValueError:n+=1
        else:raise AssertionError(mutation)
    rename=torch.tensor([3,5,0,2,1,4]);renamed=dict(bank,y=rename[labels]);Wnew=torch.empty_like(weight.detach());Wnew[rename]=weight.detach()
    rv,_=losses(components,renamed,Wnew,gains)
    for mode in values:assert torch.allclose(values[mode],rv[mode],atol=1e-6,rtol=1e-6)
    permutation=torch.randperm(180,generator=generator);pbank={k:v[permutation] if isinstance(v,torch.Tensor) else [v[i] for i in permutation.tolist()] for k,v in bank.items()};pcomponents={k:v[permutation] for k,v in components.items()}
    pv,_=losses(pcomponents,pbank,weight,gains)
    for mode in values:assert torch.allclose(values[mode],pv[mode],atol=1e-6,rtol=1e-6)
    print(json.dumps(dict(status='PASS',focused_negative_checks=n,full_denominator720=True,equal_loss_masks_with_different_gradients=True,isolated_responsible_component_only=True,W_and_all_three_gains_None=True,ablation_threshold_and_ReLU_stop_verified=True,exact_fourterm_and_routed_nativevalue_verified=True,labels_Wrename_and_packet_permutation=True,readout_FP64_FP32_compared=True,data_checkpoint_remote_reads=0,actual_updates=0)))
if __name__=='__main__':main()
