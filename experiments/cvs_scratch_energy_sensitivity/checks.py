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
    from .model import build,groups
    from .source import expected_counts
    from .calibration import calibrate
    from .warmup import head_warmup
    from experiments.cvs_e20_budget_fishr.checks import fixture_parent
    from experiments.cvs_early_geometry.checks import fixture
    from experiments.cvs_e20_peer_soft.objective import objective as previous
    from experiments.cvs_e20_peer_soft import design as parent_design
    torch.set_num_threads(2);torch.manual_seed(833);torch.backends.cudnn.deterministic=True
    configs={r['arm']:d.config(r) for r in d.rows() if r['model_seed']==d.SEEDS[0]}
    with TemporaryDirectory() as directory:_,U,bank,cache_evidence=cache_checks(configs['MARGIN_ENERGY'],Path(directory))
    L,_=fixture();gen=torch.Generator().manual_seed(551)
    U.update(x=torch.randn(1,2,256,generator=gen).expand(pc.COUNT,-1,-1),rx=L['rx'].repeat(9),day=L['day'].repeat(9))
    li=torch.tensor([i*70 for i in range(90)]+[i*70+1 for i in range(38)]);ui=torch.arange(128)
    results={}
    for arm,c in configs.items():
        with patch('torch.load',side_effect=AssertionError('Student read an initialization checkpoint')):model=build(c,device)
        assert model._parent_provenance['status']=='SCRATCH_INITIALIZATION'
        groups(model,c)
        assert bool(hasattr(model,'multilevel'))==c['multilevel']['enabled']
        with TemporaryDirectory() as directory:
            warm=head_warmup(c,model,L,device,Path(directory));assert warm['base_state_unchanged']
            assert warm['optimizer_steps']==(200 if arm=='PROBE_FISH' else 0)
            cal=calibrate(c,model,L,device,Path(directory),after_identity_updates=c['multilevel']['calibration_after_updates'])
        model.train();pg,_=groups(model,c);model.zero_grad(set_to_none=True)
        torch.manual_seed(804)
        if device.startswith('cuda'):torch.cuda.manual_seed_all(804)
        early,_,_,info=objective(c,model,None,L,li,None,None,444,device)
        torch.manual_seed(804)
        if device.startswith('cuda'):torch.cuda.manual_seed_all(804)
        plain=F.cross_entropy(model(L['x'][li].to(device)),L['y'][li].to(device));assert torch.equal(early,plain)
        calls=[];forward=model.forward_identity_only
        def count(x):calls.append(len(x));return forward(x)
        model.forward_identity_only=count
        total,parts,w,info=objective(c,model,bank if c['unlabeled'] else None,L,li,U if c['unlabeled'] else None,ui if c['unlabeled'] else None,1332,device)
        expected=(3 if c['unlabeled'] else 1)+(2 if c['multilevel']['enabled'] else 0)+(1 if c['multilevel']['mixstyle_l'] else 0)+(5 if c['multilevel']['style_selection'] in ('random','hard') else 0)
        assert len(calls)==expected,(arm,calls,expected)
        assert torch.equal(total,sum(w[k] for k in BASE_WEIGHTED_KEYS))
        torch.testing.assert_close(w['KL'],w['KL_same']+w['KL_disagree'])
        if c['multilevel']['enabled']:
            aux=torch.autograd.grad(w['aux_ce'],tuple(p for p in model.base.parameters() if p.requires_grad),retain_graph=True,allow_unused=True)
            assert all(g is None or not bool(g.any()) for g in aux)
            g=torch.autograd.grad(w['fishr'],tuple(p for p in model.base.parameters() if p.requires_grad),retain_graph=True,allow_unused=True)
            assert any(z is not None and bool(z.any()) for z in g)
            if c['multilevel']['statistic']=='probe':
                hg=torch.autograd.grad(w['fishr'],tuple(model.multilevel.heads.parameters()),retain_graph=True,allow_unused=True)
                assert all(z is None or not bool(z.any()) for z in hg)
            else:
                assert not list(model.multilevel.parameters()) and not model.multilevel.state_dict()
        if c['multilevel']['mixstyle_l']:
            assert set(info['L_style']['levels'])==(set() if info['L_style']['identity_selected'] else set(c['multilevel']['style_levels']))
            assert info['L_style']['clean_candidate_included'] and info['L_style']['candidate_pool_size']==5
            assert info['L_style']['eval_selected']['CE']>=info['L_style']['eval_clean']['CE']
            assert info['style_search_forwards']==(5 if arm!='L_REPLAY' else 0)
            assert info['L_style']['main_rng_restored'] and info['L_style']['dropout_same_mask_requested']
            if arm=='L_REPLAY':assert info['style_prediction_change']==0 and torch.equal(parts['ce'],parts['style_ce'])
            sg=torch.autograd.grad(w['style_ce'],tuple(p for p in model.base.parameters() if p.requires_grad),retain_graph=True,allow_unused=True)
            assert any(z is not None and bool(z.any()) for z in sg)
        total.backward();assert torch.isfinite(total) and all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        del model.forward_identity_only
        if arm in ('CE','SSL'):
            oc=parent_design.config(next(r for r in parent_design.rows() if r['arm']==('CE' if arm=='CE' else 'PEER_ALL') and r['model_seed']==c['model_seed']))
            values=[]
            for fn,config in ((objective,c),(previous,oc)):
                model.zero_grad(set_to_none=True);torch.manual_seed(891)
                if device.startswith('cuda'):torch.cuda.manual_seed_all(891)
                loss,*_=fn(config,model,bank if c['unlabeled'] else None,L,li,U if c['unlabeled'] else None,ui if c['unlabeled'] else None,1332,device)
                loss.backward();values.append((loss.detach(),[None if p.grad is None else p.grad.clone() for p in model.parameters()],torch.get_rng_state().clone()))
            assert torch.equal(values[0][0],values[1][0]) and torch.equal(values[0][2],values[1][2])
            assert all((a is None and b is None) or (a is not None and b is not None and torch.equal(a,b)) for a,b in zip(values[0][1],values[1][1]))
        results[arm]=dict(joint_cap=info.get('joint_gradient_cap'),forwards=expected,finite_gradients=True,head_only_CE=True,counts=expected_counts(c),calibration=cal)
    return dict(status='PASS',device=device,rows=results,cache=cache_evidence,calibration_timing_scope='bounded fixture applies calibration at declared index; production timing separately tested',target_access=False,U_truth_hidden=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--device',default='cpu');p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    value=run(a.device);d.write(a.output,value);print(json.dumps(dict(status=value['status'],device=a.device,arms=len(value['rows']),target_access=False)))
