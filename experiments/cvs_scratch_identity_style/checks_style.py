"""Actual native forwards with controlled readout fixtures for the new branch."""
import argparse,json
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch
import torch
from torch.nn import functional as F
from . import design as d,style as s
from .model import build
from .analyze_source import _style_search

def run(device='cpu'):
    torch.set_num_threads(2);torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    c=d.config(d.rows()[0]);model=build(c,device).train()
    index=torch.cat((torch.arange(30),torch.arange(2)))
    x=torch.randn(30,2,256,generator=torch.Generator().manual_seed(41133))[index].to(device)
    y=torch.arange(6).repeat_interleave(5)[index];rx=torch.tensor([1,3,4,6,8]).repeat(6)[index]
    day=torch.ones(len(index),dtype=torch.long);ids=index.tolist()
    config=dict(c['multilevel'],model_seed=c['model_seed']);step=1333
    rng=s.capture_rng(model);clean=model(x);after=s.capture_rng(model)
    state=s._snapshot(model);modes=[(m,m.training) for m in model.modules()]
    hooks=sum(len(m._forward_hooks) for m in model.modules());actual=s._metrics
    records=[]
    for name,losses,want in [('identity',[9.,1.,2.,2.,3.],-1),('style',[0.,1.,4.,4.,2.],1),('tie',[4.,1.,4.,4.,2.],1)]:
        position=[0];calls=[];forward=model.forward
        def counted(a):calls.append((model.training,torch.is_grad_enabled()));return forward(a)
        def metrics(logits,labels):
            value=actual(logits,labels);value['CE']=losses[position[0]];position[0]+=1;return value
        with patch.object(s,'_metrics',side_effect=metrics),patch.object(model,'forward',side_effect=counted):
            logits,info=s.styled_forward(model,x,y,rx,day,ids,step,config,1.,rng)
        assert position[0]==5 and calls==[(False,False)]*5+[(True,True)]
        assert info['selected_index']==want and info['identity_selected']==(want==-1)
        assert info['eval_selected']['CE']==max(losses) and info['total_extra_forwards']==6
        assert info['selected_levels']==([] if want==-1 else list(s.LEVELS))
        assert bool(info['levels'])==(want!=-1) and info['actual_hook_order']==info['selected_levels']
        if want==-1:
            torch.testing.assert_close(logits,clean,rtol=0,atol=0)
            assert info['changed_packets']==0 and info['selected_seed'] is None
        else:assert info['changed_packets']>0
        gradient=torch.autograd.grad(F.cross_entropy(logits,y.to(device)),tuple(p for p in model.base.parameters() if p.requires_grad),allow_unused=True)
        assert any(g is not None and bool(g.abs().sum()>0) for g in gradient)
        assert all(g is None or bool(torch.isfinite(g).all()) for g in gradient)
        assert s._same_rng(after,s.capture_rng(model)) and s._same_state(model,state)
        assert all(m.training==flag for m,flag in modes)
        assert sum(len(m._forward_hooks) for m in model.modules())==hooks
        cfg=deepcopy(c);cfg['batch_size']=len(y)
        checked=_style_search(info,cfg,step,1.)
        assert checked['identity_selected']==float(want==-1)
        records.append(dict(case=name,details=info,finite_nonzero_identity_gradient=True))
    # The unmodified actual readouts also obey max(clean, four whole styles).
    logits,real=s.styled_forward(model,x,y,rx,day,ids,step,config,1.,rng)
    assert real['eval_selected']['CE']==max([real['eval_clean']['CE']]+real['candidate_losses'])
    _style_search(real,cfg,step,1.)
    rejects=[]
    for key,value in [('clean_candidate_included',False),('identity_selected',False),('selected_index',0),('selected_seed',123),('candidate_pool_size',4)]:
        bad=deepcopy(records[0]['details']);bad[key]=value
        try:_style_search(bad,cfg,step,1.)
        except ValueError:rejects.append(key)
        else:raise AssertionError('Inconsistent identity replay accepted: '+key)
    for bad in [dict(config,include_identity_candidate=False),dict(config,source_role='U_s'),dict(config,style_selection='random')]:
        try:s.styled_forward(model,x,y,rx,day,ids,step,bad,1.,rng)
        except ValueError:pass
        else:raise AssertionError('Unregistered selection/source accepted')
    # A selected-clean training failure must unwind all modes and RNG state.
    position=[0];forward=model.forward
    def identity_metrics(logits,labels):
        v=actual(logits,labels);v['CE']=10. if position[0]==0 else 1.;position[0]+=1;return v
    def failure(a):
        if model.training:raise RuntimeError('intentional identity replay failure')
        return forward(a)
    with patch.object(s,'_metrics',side_effect=identity_metrics),patch.object(model,'forward',side_effect=failure):
        try:s.styled_forward(model,x,y,rx,day,ids,step,config,1.,rng)
        except RuntimeError as e:assert str(e)=='intentional identity replay failure'
        else:raise AssertionError('Selected-clean failure not propagated')
    assert s._same_rng(after,s.capture_rng(model)) and s._same_state(model,state)
    assert all(m.training==flag for m,flag in modes) and model not in s._ACTIVE
    assert sum(len(m._forward_hooks) for m in model.modules())==hooks
    return dict(status='PASS',device=device,source_only_synthetic=True,target_access=False,records=records,actual_readout=real,negative_checks=rejects,identity_replay_exception_cleanup=True,actual_forward_budget=6,fixture_readout_overrides='only controlled branch fixtures replace CE scalar; actual native forwards/backward retained; one additional real-readout test')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--device',default='cpu');p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    v=run(a.device);d.write(a.output,v);print(json.dumps(dict(status=v['status'],device=a.device,branches=3,target_access=False)))
