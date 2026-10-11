import argparse,copy,json,tempfile
from pathlib import Path
import torch
from . import design as d
from .model import OriginalIdentity,CurrentStudentTarget,make_target
from experiments.cvs_report_ssl.model import ReportIdentity,groups


def check():
    from experiments.cvs_stage2_decouple.design import make_args
    from experiments.cvs_phase1_stack.runtime import installed
    from experiments.cvs_equivariant_identity.precision import numerical_context
    from .source import objective,U_diagnostic
    from experiments.cvs_report_ssl.auxiliary import PhysicalHeads
    pc=d.original.parent_config(d.SEEDS[0]);a=make_args(pc,'cpu');a.num_domains,a.input_len=5,256
    with numerical_context(d.FULL_FP32_POLICY),installed(pc) as native:base=native.build_baseline_model(a,torch.device('cpu'))
    base._stage2_baseline_args=vars(a);base.eval();x=torch.randn(16,2,256);y=torch.arange(16)%6
    with torch.no_grad():original_logits=base.forward_identity_only(x)['tx_logits']
    records=[]
    for row in [r for r in d.rows() if r['model_seed']==d.SEEDS[0]]:
        c=d.config(row);d.validate(c);model=OriginalIdentity(copy.deepcopy(base)) if c['reference_mode']=='original' else ReportIdentity(copy.deepcopy(base),False)
        if c['reference_mode']=='original':
            model.eval()
            with torch.no_grad():assert torch.equal(original_logits,model(x))
        pg,_=groups(model,c);model.train();heads=PhysicalHeads(False,False)
        target=make_target(c,model) if c['unlabeled'] else None
        state={n:p.detach().clone() for n,p in model.named_parameters()};modes={n:m.training for n,m in model.named_modules()}
        if target:
            q=target.probabilities(x);assert not q.requires_grad and q.shape==(16,6)
            assert modes=={n:m.training for n,m in model.named_modules()}
            if not c['teacher']:assert target.model is model and target.updates==0
            scan=U_diagnostic(target,dict(x=x,rx=torch.ones(16,dtype=torch.long),day=torch.ones(16,dtype=torch.long)),'cpu')
            assert scan['count']==16 and scan['U_accuracy'] is None and len(scan['recomputed_state40_mean'])==40
        L=dict(x=x,y=y);U=dict(x=x.clone());idx=torch.arange(16)
        loss,parts,weighted,details=objective(c,model,heads,L,idx,U,idx,target,None,889,'cpu')
        loss.backward();assert torch.isfinite(loss)
        assert all(torch.equal(p,state[n]) for n,p in model.named_parameters())
        if target:target.update(model)
        if target and not c['teacher']:assert target.updates==0 and target.model is model
        records.append(dict(arm=c['arm'],reference=c['reference_mode'],target=c['semi']['target_provider'],encoder_peak=pg[0]['peak'],head_peak=pg[1]['peak'],target_owned_copy=bool(c['teacher']),finite_loss=True))
    assert len(d.rows())==24 and all(d.config(r)['updates']==8880 for r in d.rows())
    return dict(status='PASS',rows=records,original_logits_exact=True,live_target_no_copy_or_update=True,weak_forward_restores_modes=True,finite_gradients=True,identity_updates=0,target_access=False)


def remote_preflight():
    from .model import build
    from .source import load_source
    result=check();sources=[]
    for seed in d.SEEDS:
        c=d.config(next(r for r in d.rows() if r['model_seed']==seed and r['arm']=='O_DIRECT'))
        L,U,V,contract,p=load_source(c);model=build(c,'cpu').eval()
        from experiments.cvs_stage2_decouple.common import build_model
        original=build_model(c['parent_contract'],'cpu').eval()
        with torch.no_grad():
            x=L['x'][:16];assert torch.equal(model(x),original.forward_identity_only(x)['tx_logits'])
            t=CurrentStudentTarget(model);assert torch.equal(t.probabilities(x),model(x).softmax(1))
        sources.append(dict(seed=seed,parent_provenance=p,L=len(L['ids']),U=len(U['ids']),V=len(V['ids']),original_parent_logits_exact=True,U_truth_hidden=True))
    result.update(sources=sources)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--remote-preflight',action='store_true');p.add_argument('--output',required=True);a=p.parse_args();torch.set_num_threads(2)
    value=remote_preflight() if a.remote_preflight else check();d.write(a.output,value);print(json.dumps(value))
