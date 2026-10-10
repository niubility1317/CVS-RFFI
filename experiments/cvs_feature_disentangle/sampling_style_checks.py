"""Focused formula, role, RNG, relationship and checkpoint tests."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from torch import nn
from .sampling import RelationStream
from .style import WeakConditionalMixStyle


class Fixture:
    def __init__(self,n=9):
        self.index=[]
        for y in range(6):
            for rx in (1,3,4,6,8):
                for day in (0,1,2):
                    for sig in range(n): self.index.append(SimpleNamespace(tx_i=y,rx_i=rx,day_i=day,eq_i=0,sig_i=sig))
    def __getitem__(self,i):
        r=self.index[i]
        return torch.full((2,16),float(i)),r.tx_i,r.rx_i,dict(rx_i=r.rx_i,day_i=r.day_i,eq_i=0,sig_i=r.sig_i,base_index=i)


def check(device):
    tests=[]; torch_state=torch.random.get_rng_state().clone(); np_state=np.random.get_state()
    cuda_state=torch.cuda.get_rng_state(device).clone() if device.type=='cuda' else None
    ds=Fixture(); stream=RelationStream(ds,7)
    for k in range(30):
        x,y,rx,meta=stream.next_batch((k+1)*8)
        assert x.shape==(48,2,16) and len(set(meta['base_index'].tolist()))==48
        assert len(meta['day_i'].unique())==1 and len(rx.unique())==2
        assert all(int(((y==label)&(rx==r)).sum())==4 for r in rx.unique() for label in range(6))
    assert stream.summary()['blocks_covered']==30 and set(stream.coverage.values())=={1}
    tests.append('all_30_blocks_one_cycle_balanced_unique_B48')
    state=stream.state_dict(); next1=stream.next_indices(248)
    stream.load_state_dict(state); assert stream.next_indices(248)==next1
    tests.append('sampler_checkpoint_exact_replay')
    # Cell size nine: prove first nine draws are unique even with 4-sized batches.
    isolated=RelationStream(ds,8); key=(0,1,0)
    draws=[i for _ in range(3) for i in isolated._take(key,isolated.cells[key],4)]
    assert len(set(draws[:9]))==9
    tests.append('cell_exhaustion_before_reshuffle_nonmultiple_of_four')
    bad=Fixture(3); empty=RelationStream(bad,9)
    assert empty.next_batch(8) is None and empty.summary()['insufficient_cells']
    tests.append('insufficient_cells_skipped_no_duplication')
    random=RelationStream(ds,10,mode='random')
    assert len(set(random.next_indices(8)))==48
    tests.append('random_B48_equal_exposure')
    # No global RNG is consumed by sampler or style, including beta draws.
    h=torch.arange(6*3*8,device=device,dtype=torch.float64).reshape(6,3,8)/80
    h.requires_grad_(); metadata=dict(y=[0,0,0,1,1,1],rx=[1,3,4,1,3,4],day=[0,0,1,0,0,0],pid=list(range(6)),role='native_L_clean',epoch=40)
    s=WeakConditionalMixStyle(23)
    assert s.probability(20)==0 and s.probability(30)==.075 and s.probability(40)==.15 and s.probability(200)==.15
    for _ in range(100):
        with s.context(**metadata): out=s.apply(h)
        if s.last['applied']: break
    assert s.last['applied']; detail=s.last
    assert detail['a'][2]==0 and torch.equal(out[2],h[2])
    for i,j in enumerate(detail['donors']):
        if detail['a'][i]>0: assert metadata['y'][i]==metadata['y'][j] and metadata['rx'][i]!=metadata['rx'][j] and metadata['day'][i]==metadata['day'][j] and metadata['pid'][i]!=metadata['pid'][j]
    assert max(detail['a'])<=.2
    mu=h.detach().mean(-1,keepdim=True); std=(h.detach().var(-1,unbiased=False,keepdim=True)+s.eps).sqrt()
    blend=torch.tensor(detail['a'],device=device,dtype=h.dtype)[:,None,None]; donor=detail['donors']
    expected=(h-mu)/std*((1-blend)*std+blend*std[donor])+(1-blend)*mu+blend*mu[donor]
    assert torch.allclose(out,expected,atol=1e-10)
    tests.append('legal_sameTX_diffRX_sameDay_diffPID_donor_weak_bound')
    # Analytic derivative has no cross-sample donor path because both moments detach.
    grad=torch.autograd.grad(out[0].sum(),h,retain_graph=True)[0]
    sigma=(h.detach().var(-1,unbiased=False,keepdim=True)+s.eps).sqrt()
    a=detail['a'][0]; j=detail['donors'][0]
    expected=((1-a)*sigma[0]+a*sigma[j])/sigma[0]
    assert torch.allclose(grad[0],expected.expand_as(grad[0]),atol=1e-10)
    assert torch.count_nonzero(grad[1:])==0
    tests.append('statistics_detach_analytic_gradient_no_donor_gradient')
    native_count=s.snapshot()['roles']['native_L_clean']['calls']
    for role in ('validation','D_fit','D_calibration','Fishr','U','EMA_teacher','real_action','relation_CE'):
        with s.context(**dict(metadata,role=role)): assert s.apply(h) is h
    assert s.snapshot()['roles']['native_L_clean']['calls']==native_count
    assert s.apply(h) is h and s._context is None
    tests.append('disabled_roles_context_clear_accumulation_preserved')
    no=WeakConditionalMixStyle(25)
    for _ in range(60):
        with no.context(**dict(metadata,rx=[1]*6)): assert torch.equal(no.apply(h),h)
    assert no.snapshot()['roles']['native_L_clean']['donor_coverage']==0
    tests.append('no_donor_identity_no_crossTX_fallback')
    class Model(nn.Module):
        def __init__(self): super().__init__(); self.time_down=nn.Identity(); self.t1=nn.Identity()
        def forward(self,x): return self.t1(self.time_down(x))
    model=Model(); hook=WeakConditionalMixStyle(28).attach(model)
    with hook.context(**metadata): model(h)
    assert hook.snapshot()['roles']['native_L_clean']['calls']==1
    with hook.context(**metadata): model(h); model(h)
    assert hook.snapshot()['roles']['unscoped']['calls']==1
    hook.detach(); assert not model.time_down._forward_hooks and not model.t1._forward_hooks
    dup=nn.ModuleDict({'a':Model(),'b':Model()})
    try: WeakConditionalMixStyle(28).attach(dup)
    except ValueError: pass
    else: raise AssertionError('Duplicate time_down accepted')
    tests.append('unique_time_down_hook_no_context_reuse')
    assert torch.equal(torch_state,torch.random.get_rng_state())
    if cuda_state is not None: assert torch.equal(cuda_state,torch.cuda.get_rng_state(device))
    now=np.random.get_state(); assert np_state[0]==now[0] and np.array_equal(np_state[1],now[1]) and np_state[2:]==now[2:]
    tests.append('global_torch_numpy_rng_unchanged')
    return dict(status='PASS',device=str(device),checks=tests,relation_summary=stream.summary(),style=s.snapshot())


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--device',default='cpu'); parser.add_argument('--output'); args=parser.parse_args()
    result=check(torch.device(args.device)); content=json.dumps(result,indent=2)
    if args.output:
        path=Path(args.output); path.parent.mkdir(parents=True,exist_ok=True); path.write_text(content,encoding='utf-8')
    print(content)
