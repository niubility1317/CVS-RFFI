"""Bounded source-only test and strict source teacher inheritance check."""
import argparse,tempfile
from pathlib import Path
from unittest.mock import patch
import torch
from . import design as d
from .model import build,groups
from .source import expected_counts
from .teacher import expected_proof,check_proof

def cache_checks(c,root):
    from .peer_cache import load,SCHEMA
    gen=torch.Generator().manual_seed(915);ids=[f'synthetic-U-{i}' for i in range(56700)]
    U=dict(ids=ids,role='U_s');root=Path(root);root.mkdir(exist_ok=True)
    for seed in d.SEEDS:
        logits=torch.randn(56700,6,generator=gen);q=logits.softmax(1)
        torch.save(dict(schema=SCHEMA,seed=seed,source_role='U_s',ids=ids,probabilities=q,provenance=expected_proof(seed)),root/f'seed-{seed}.pt')
    d.write(root/'completion.json',dict(schema=SCHEMA,status='COMPLETE',source_role='U_s',count=56700,seeds=list(d.SEEDS),target_access=False,U_truth=False,provenance={str(s):expected_proof(s) for s in d.SEEDS}))
    from copy import deepcopy
    test=deepcopy(c);test['teacher']['cache_root']=str(root);bank=load(test,U)
    try:load(test,dict(U,y=torch.zeros(56700)))
    except ValueError:pass
    else:raise AssertionError('U truth accepted')
    try:check_proof(dict(expected_proof(d.SEEDS[0]),explicit_CFO_identity_input=True),d.SEEDS[0])
    except ValueError:pass
    else:raise AssertionError('Historical CFO teacher accepted')
    return test,U,bank,dict(status='PASS',synthetic=True,real_data=False)

def local():
    records=[];torch.set_num_threads(2)
    from .provenance import expected_sources
    from .evaluate import checkpoint_provenance
    for row in d.rows():
        c=d.validate(d.config(row));count=expected_counts(c)
        assert count['identity_updates']==4440 and count['U_packets']==511488 and count['auxiliary_steps']==0
        assert count['clean_replay_steps']==0 and count['relation_steps']==999 and count['relation_L']==29970 and count['relation_forwards']==999
        records.append(dict(row_id=row['row_id'],counts=count))
    for arm in d.ARMS:
        c=d.config(next(r for r in d.rows() if r['arm']==arm))
        with patch('torch.load',side_effect=AssertionError('Initial weights loaded')):model=build(c,'cpu')
        groups(model,c);model.eval();teachers={str(s):expected_proof(s) for s in c['teacher']['seeds']}
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'own.pt';payload=dict(model=model.state_dict(),baseline_args=model._stage2_baseline_args,config=c,model_seed=c['model_seed'],epoch=20,total_epochs=20,parent_training_epochs=0,incremental_epochs=20,optimizer_steps=4440,parent_checkpoint=None,checkpoint_selection='final_only',target_access=False,run_id=c['run_id'],candidate_id=c['row_id'],parent_provenance=model._parent_provenance,scratch_only=True,scratch_scope='student_identity_only',reference_E20_loaded=False,teacher_provenance=teachers,checkpoint_sources=expected_sources(c))
            torch.save(payload,p);restored=build(c,'cpu',p).eval();checkpoint_provenance(c,payload)
            with torch.no_grad():assert torch.equal(model(torch.zeros(2,2,256)),restored(torch.zeros(2,2,256)))
            torch.save(dict(payload,teacher_provenance={}),p)
            try:build(c,'cpu',p)
            except ValueError:pass
            else:raise AssertionError('Missing teacher ancestry accepted')
    return dict(status='PASS',rows=records,strict_own_checkpoint_roundtrip=True,teachers_no_explicit_CFO=True,target_access=False)

def preflight():
    from .source import load_source
    from .objective import objective
    from .teacher import load
    from .update import take_update
    from experiments.cvs_equivariant_identity.precision import numerical_context
    result=local();spec=d.read(d.ROOT/'registration/experiment.json');assert [r['config'] for r in spec['rows']]==[d.config(r) for r in d.rows()]
    c=d.config(d.rows()[0]);L,U,V,contract=load_source(c);assert 'y' not in U
    probabilities={};proofs={};idx=torch.arange(32)
    with numerical_context(d.FULL_FP32_POLICY):
        for seed in d.SEEDS:
            model,proof=load(seed,'cpu')
            with torch.no_grad():probabilities[seed]=model(U['x'][idx]).softmax(1)
            proofs[str(seed)]=proof
        evidence=[]
        for arm in d.ARMS:
            c=d.config(next(r for r in d.rows() if r['arm']==arm));a,b=[probabilities[s] for s in c['teacher']['seeds']];q=(a+b)/2
            bank=dict(q=q,ids=U['ids'][:32],teacher_seeds=c['teacher']['seeds'],peer_disagreement=a.argmax(1)!=b.argmax(1),peer_JS=.5*((a*(a.clamp_min(1e-12).log()-q.clamp_min(1e-12).log())).sum(1)+(b*(b.clamp_min(1e-12).log()-q.clamp_min(1e-12).log())).sum(1)))
            model=build(c,'cpu');pg,_=groups(model,c);model.train();optimizer=torch.optim.AdamW(pg,weight_decay=c['optimizer']['weight_decay'])
            loss,_,w,detail,_,norm,update=take_update(c,model,optimizer,lambda:objective(c,model,bank,L,idx,U,idx,1332,'cpu'))
            assert detail['relation']['active'] and detail['relation']['L_packets']==30 and detail['relation']['support_query_PID_disjoint']
            assert torch.isfinite(loss) and detail['U_packets']==32 and not model._identity_input_policy['explicit_CFO_identity_input'] and update['sam_applied']==c['engineering']['sam']
            assert all(int(v['step'])==1 for v in optimizer.state.values()) and torch.allclose(w['KL'],w['KL_same']+w['KL_disagree'],atol=1e-6)
            evidence.append(dict(arm=arm,actual_L=len(L['ids']),actual_U=len(U['ids']),V_backward=False,real_L_U_backward=True,no_explicit_CFO=True,relation=detail['relation'],U_truth=False,gradient_norm=norm,SAM=update['sam_applied']))
    return dict(result,actual_source_evidence=evidence,teacher_provenance=proofs)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--remote-preflight',action='store_true');a=p.parse_args();d.write(a.output,preflight() if a.remote_preflight else local())
