"""Native bounded teacher, KL routing and strict cache regression checks."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import torch
from torch.nn import functional as F
from . import design as d
from . import peer_cache as pc
from .objective import objective,BASE_WEIGHTED_KEYS,AUDIT_ONLY_KEYS


def cache_checks(c,root):
    from experiments.cvs_e20_foundation.verify import config,EXECUTION_COMMIT
    ids=[f'U-{i}' for i in range(pc.COUNT)]
    U=dict(role='U_s',ids=ids)
    proofs={};payloads={}
    for index,seed in enumerate(pc.SEEDS):
        source=config(seed)
        proof=dict(status='VERIFIED',seed=seed,epoch=20,optimizer_steps=4440,scratch_only=True,ancestors=[],source_roles='EXACT_MATCH',
            training_target_access=False,auxiliary_inheritance=False,execution_commit=EXECUTION_COMMIT,source_run=source['run_id'],
            checkpoint=str(Path(source['output_root'])/'final_ssdg.pth'),checkpoint_selection='fixed final20; all original matrix rows evaluated',
            selection_context='synthetic test only; exposed-baseline context retained')
        labels=(torch.arange(pc.COUNT)+(torch.arange(pc.COUNT)%2)*index)%6
        q=F.one_hot(labels,6).float()*.988+.002
        order=torch.arange(pc.COUNT-1,-1,-1) if index==1 else torch.arange(pc.COUNT)
        payload=dict(schema=pc.SCHEMA,seed=seed,source_role='U_s',ids=[ids[i] for i in order.tolist()],probabilities=q[order],provenance=proof)
        proofs[str(seed)]=proof;payloads[seed]=payload;torch.save(payload,root/f'seed-{seed}.pt')
    completion=dict(schema=pc.SCHEMA,status='COMPLETE',source_role='U_s',count=pc.COUNT,seeds=list(pc.SEEDS),target_access=False,U_truth=False,provenance=proofs)
    (root/'completion.json').write_text(json.dumps(completion),encoding='utf-8')
    c=deepcopy(c);c['teacher']['cache_root']=str(root);bank=pc.load(c,U)
    a=payloads[pc.SEEDS[1]]['probabilities'].flip(0);b=payloads[pc.SEEDS[2]]['probabilities']
    assert torch.equal(bank['q'],(a+b)/2) and bank['ids']==ids
    assert torch.equal(bank['peer_disagreement'],a.argmax(1)!=b.argmax(1))
    assert not bank['q'].requires_grad and all(seed!=c['model_seed'] for seed in bank['teacher_seeds'])
    assert set(bank['provenance'])=={str(s) for s in bank['teacher_seeds']}
    rejects=[]
    def reject(name,call):
        try:call()
        except (ValueError,KeyError,FileNotFoundError):rejects.append(name)
        else:raise AssertionError('Invalid cache accepted: '+name)
    reject('U truth',lambda:pc.load(c,dict(U,y=object())))
    reject('U role',lambda:pc.load(c,dict(U,role='V')))
    reject('repeated U PID',lambda:pc.load(c,dict(U,ids=[ids[0]]+ids[:-1])))
    bad=deepcopy(c);bad['teacher']['seeds']=[c['model_seed'],pc.SEEDS[1]]
    reject('student included',lambda:pc.load(bad,U))
    bad=deepcopy(completion);bad['status']='WRITING';(root/'completion.json').write_text(json.dumps(bad),encoding='utf-8')
    reject('incomplete manifest',lambda:pc.load(c,U));(root/'completion.json').write_text(json.dumps(completion),encoding='utf-8')
    seed=pc.SEEDS[1];file=root/f'seed-{seed}.pt';original=payloads[seed]
    def corrupted(name,mutator):
        value=deepcopy(original);mutator(value);torch.save(value,file);reject(name,lambda:pc.load(c,U));torch.save(original,file)
    corrupted('cache PID mismatch',lambda v:v['ids'].__setitem__(0,'not-U'))
    corrupted('cache duplicated PID',lambda v:v['ids'].__setitem__(0,v['ids'][1]))
    corrupted('cache truth field',lambda v:v.__setitem__('y',torch.zeros(pc.COUNT)))
    corrupted('cache role pollution',lambda v:v.__setitem__('source_role','V'))
    corrupted('nonfinite probability',lambda v:v['probabilities'].__setitem__((0,0),float('nan')))
    corrupted('non-simplex probability',lambda v:v['probabilities'].__setitem__((0,0),.5))
    corrupted('incorrect shape',lambda v:v.__setitem__('probabilities',torch.ones(12,6)/6))
    corrupted('contaminated teacher',lambda v:v['provenance'].__setitem__('training_target_access',True))
    corrupted('wrong parent',lambda v:v['provenance'].__setitem__('checkpoint','other.pth'))
    assert pc.load(dict(c,teacher=dict(c['teacher'],source='none')),None) is None
    return c,U,bank,dict(PID_reordered_exactly=True,self_teacher_excluded=True,provenance_keyed_by_teacher_seed=True,finite_simplex_and_source_proof_checked=True,rejected_cases=rejects)


def run(device='cpu'):
    torch.set_num_threads(2);torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    configs={r['arm']:d.config(r) for r in d.rows() if r['model_seed']==d.SEEDS[0]}
    for c in configs.values():d.validate(c)
    Path('local_artifacts').mkdir(exist_ok=True)
    with TemporaryDirectory(prefix='peer_cache_check_',dir='local_artifacts') as directory:
        config,U,bank,cache_evidence=cache_checks(configs['PEER_ALL'],Path(directory))
    from experiments.cvs_early_geometry import design as old_design
    from experiments.cvs_early_geometry.model import build
    from experiments.cvs_e20_routed_fishr.multilevel import MultiLevel
    from experiments.cvs_e20_routed_fishr.objective import objective as old_objective
    from experiments.cvs_e20_routed_fishr import design as old_routed
    from experiments.cvs_e20_budget_fishr.multilevel_checks import duplicate_pid_checks
    from experiments.cvs_early_geometry.objective import weak
    model=build(old_design.config(next(r for r in old_design.rows() if r['arm']=='PHYS')),device)
    model.multilevel=MultiLevel(model.base,seed=d.SEEDS[0]).to(device);model.train()
    gen=torch.Generator().manual_seed(4022)
    L=dict(x=torch.randn(180,2,256,generator=gen),y=torch.arange(6).repeat_interleave(10).repeat(3),rx=torch.tensor([1,3,4,6,8]).repeat_interleave(2).repeat(18),day=torch.arange(3).repeat_interleave(60),ids=list(range(180)))
    U.update(x=torch.randn(1,2,256,generator=gen).expand(pc.COUNT,-1,-1),rx=L['rx'].repeat(315),day=L['day'].repeat(315))
    class Hidden(dict):
        def __getitem__(self,key):
            if key=='y':raise AssertionError('U truth read')
            return super().__getitem__(key)
    U=Hidden(U);li=torch.arange(128);ui=torch.tensor(list(range(127))+[0]);step=1332
    initial={n:v.clone() for n,v in model.state_dict().items()}
    def seed():
        torch.manual_seed(478)
        if torch.cuda.is_available():torch.cuda.manual_seed_all(478)
    def rng():return [torch.random.get_rng_state().clone()]+(torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])
    def execute(function,c,teacher):
        model.load_state_dict(initial);model.zero_grad(set_to_none=True);seed()
        total,parts,weighted,detail=function(c,model,teacher,L,li,U,ui,step,device)
        total.backward()
        grads={n:None if p.grad is None else p.grad.clone() for n,p in model.named_parameters()}
        return float(total.detach()),{k:float(v.detach()) for k,v in parts.items()},grads,rng(),detail
    old=old_routed.config(next(r for r in old_routed.rows() if r['arm']=='SSL' and r['model_seed']==d.SEEDS[0]))
    a=execute(old_objective,old,None);b=execute(objective,configs['SELF_GATE'],None)
    assert a[:2]==b[:2]
    assert all((a[2][n] is None and b[2][n] is None) or (a[2][n] is not None and b[2][n] is not None and torch.equal(a[2][n],b[2][n])) for n in a[2])
    assert all(torch.equal(x,y) for x,y in zip(a[3],b[3]))
    findings={}
    params=tuple(p for n,p in model.named_parameters() if 'cls_head' in n)
    for arm,c in configs.items():
        model.load_state_dict(initial);model.zero_grad(set_to_none=True);seed()
        peer=bank if c['teacher']['source']=='peer' else None
        total,parts,w,info=objective(c,model,peer,L,li,U if c['unlabeled'] else None,ui if c['unlabeled'] else None,step,device)
        assert torch.equal(total,sum(w[k] for k in BASE_WEIGHTED_KEYS))
        torch.testing.assert_close(w['KL'],w['KL_same']+w['KL_disagree'],atol=1e-6,rtol=1e-6)
        g1=torch.autograd.grad(w['KL'],params,retain_graph=True,allow_unused=True)
        g2=torch.autograd.grad(w['KL_same']+w['KL_disagree'],params,retain_graph=True,allow_unused=True)
        for x,y in zip(g1,g2):
            if x is not None:torch.testing.assert_close(x,y,atol=2e-6,rtol=2e-5)
        if c['unlabeled']:
            assert info['weak_forwards']==1 and info['teacher_forwards']==0
            if c['teacher']['selection']=='all':assert info['accepted_U']==128
            if c['teacher']['source']=='peer':assert info['peer_disagreement']>=0 and info['peer_JS']>=-1e-6
            if c['multilevel']['mixstyle_u']:
                assert info['U_style']['high_confidence']==int((bank['q'][ui].max(1).values>=.95).sum())
                assert info['accepted_U']==128 and info['U_style']['high_confidence']<128
        else:assert info['weak_forwards']==0 and info['U_packets']==0
        total.backward()
        assert all(torch.equal(initial[n],t) for n,t in model.state_dict().items())
        findings[arm]=dict(accepted_U=info['accepted_U'],weak_forwards=info['weak_forwards'],KL=float(parts['KL'].detach()),KL_same=float(w['KL_same'].detach()),KL_disagree=float(w['KL_disagree'].detach()))
    # Per-peer diagnostics may consume RNG internally but must not affect the
    # teacher, main strong dropout, or its loss/gradients.
    baseline=execute(objective,configs['PEER_ALL'],bank)
    def noisy_diagnostic(m,x):
        result=weak(m,x);torch.rand(5,device=device);return result
    with patch('experiments.cvs_e20_peer_soft.objective.weak',side_effect=noisy_diagnostic):changed=execute(objective,configs['PEER_ALL'],bank)
    assert baseline[:2]==changed[:2] and all(torch.equal(x,y) for x,y in zip(baseline[3],changed[3]))
    for n in baseline[2]:
        x,y=baseline[2][n],changed[2][n]
        assert (x is None and y is None) or (x is not None and y is not None and torch.equal(x,y))
    return dict(status='PASS',device=device,target_access=False,cache=cache_evidence,SELF_GATE_old_SSL_bitwise_loss_gradient_RNG=True,
        KL_full_batch_denominator=True,KL_same_disagree_loss_and_gradient_sum=True,audit_keys_not_double_counted=True,
        peer_live_diagnostic_RNG_isolated=True,peer_cached_teacher_no_forward=True,ALL_KL_does_not_remove_style_q095_gate=True,
        U_truth_hidden=True,model_state_unchanged=True,repeated_PID=duplicate_pid_checks(device),arms=findings)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--device',default='cpu');parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();value=run(args.device);args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in value.items() if k not in ('arms','cache','repeated_PID')},allow_nan=False))
