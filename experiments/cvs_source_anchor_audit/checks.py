import argparse,copy,json,time
import torch
from . import design as d
from .graph import fit,predict,probability

def check():
    settings=d.config(d.rows()[0])['graph'].copy();settings.update(k=3,tolerance=1e-7,max_iterations=200)
    L=torch.eye(6);y=torch.arange(6);U=torch.cat((L+.03,L-.02));q=torch.full((12,6),.1);q[torch.arange(12),(torch.arange(12)+1)%6]=.5
    state=fit(L,y,U,q,settings);before=state['distribution'].clone()
    out=predict(state,L,q[:6]);assert all(v.shape==(6,6) for v in out.values())
    assert torch.equal(before,state['distribution']) and torch.equal(before[:6],torch.eye(6))
    # Independent dense solution to the Dirichlet system verifies the iterative solver.
    from .graph import weights,neighbors
    bank=state['bank'];idx,sim=neighbors(bank,bank,3,True);W=torch.zeros(18,18)
    for i in range(18):
        for j,w in zip(idx[i].tolist(),weights(sim[i],.07).tolist()):W[i,j]+=w/2;W[j,i]+=w/2
    W/=W.sum(1).mean();D=W.sum(1);restart=q.max(1).values
    answer=torch.linalg.solve(torch.diag(D[6:]+restart)-W[6:,6:],W[6:,:6]@torch.eye(6)+restart[:,None]*q)
    assert torch.allclose(answer,state['distribution'][6:],atol=2e-6)
    for bad in (q*2,torch.full_like(q,float('nan'))):
        try:probability(bad)
        except ValueError:pass
        else:raise AssertionError('Malformed probabilities accepted')
    assert state['diagnostics']['U_nodes_without_L_component']==0
    from .run import metrics,geometry
    base=torch.tensor([[.8,.2],[.2,.8],[.8,.2],[.2,.8]])
    corrected=torch.tensor([[.8,.2],[.8,.2],[.2,.8],[.2,.8]])
    labels=torch.tensor([0,0,0,1]);strata=torch.tensor([0,0,1,1])
    measured=metrics({'model':base,'candidate':corrected},labels,strata,strata)['candidate']
    assert measured['paired']['fix_count']==1 and measured['paired']['break_count']==1 and measured['paired']['net_corrections']==0
    assert measured['paired']['disagreement_count']==2
    identical=geometry(torch.eye(4),base,torch.eye(4),base)
    assert identical['prediction_disagreement_count']==0 and identical['normalized_feature_L2_mean']==0
    return dict(status='PASS',dense_Dirichlet_solution=True,query_cannot_update_graph=True,L_anchors_exact=True,malformed_probability_rejected=True,paired_fix_break_counts=True,identical_geometry=True,target_access=False)

def remote_preflight():
    from .run import verify_source,extract
    from .gradients import audit
    from experiments.cvs_report_ssl.source import load_source
    from experiments.cvs_report_ssl.model import build,Teacher
    from experiments.cvs_report_ssl.auxiliary import PhysicalHeads
    records=[]
    for row in d.rows():
        c=d.config(row);aux=verify_source(c);L,U,V,contract,provenance=load_source(c['source_config'])
        m=build(c['source_config'],'cpu',checkpoint=c['source_checkpoint']).eval();t=Teacher(m,.99);t.model.load_state_dict(aux['teacher'],strict=True)
        h=PhysicalHeads(c['source_config']['enabled']['linear'],False);h.load_state_dict(aux['heads'],strict=True)
        tiny=copy.deepcopy(c);tiny['gradient']['batches']=1
        result=audit(tiny,m,t,h,L,U,'cpu');assert result['virtual_steps_restored_exactly']
        assert {'U_EMA','U_student'}<=set(result['rows'][0]['tasks'])
        from .gradients import StudentSnapshot
        snapshot=StudentSnapshot(m)
        with torch.no_grad():assert torch.equal(snapshot.probabilities(L['x'][:8]),m(L['x'][:8]).softmax(1))
        records.append(dict(row_id=row['row_id'],parent_status=provenance['status'],native_gradient_batch=True,source_model_restored=True,identity_updates=0))
    return dict(status='PASS',rows=records,math=check(),target_access=False,identity_updates=0)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--remote-preflight',action='store_true');p.add_argument('--output',required=True);a=p.parse_args()
    torch.set_num_threads(2);value=remote_preflight() if a.remote_preflight else check();d.write(a.output,value);print(json.dumps(value))
