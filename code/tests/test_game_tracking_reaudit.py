import sys
from pathlib import Path
from types import SimpleNamespace
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from cvsrffi.game_tracking.data import SourceData
from cvsrffi.game_tracking.runtime import update_pseudo_problem
from cvsrffi.game_tracking.solvers import GameSolver
from cvsrffi.game_tracking.step_context import temporal_mask
from cvsrffi.game_tracking.data import SyntheticRole


def test_u_epoch_rotation_covers_pool_and_preserves_neighbors():
    data=SourceData(None,list(range(103)),None,{}, {})
    batches=[list(data.unlabeled_epoch_loader(8,epoch_index=e,steps=2,seed=392005)) for e in range(7)]
    seen={int(v) for epoch in batches for batch in epoch for v in batch}
    assert seen==set(range(103))
    assert all(bool(((batch[1:]-batch[:-1])==1).all()) for epoch in batches for batch in epoch)
    again=list(data.unlabeled_epoch_loader(8,epoch_index=4,steps=2,seed=392005))
    assert all(torch.equal(x,y) for x,y in zip(again,batches[4]))


def test_optimistic_pseudo_history_only_resets_on_large_change():
    model=torch.nn.Linear(1,1)
    opt=torch.optim.SGD(model.parameters(),lr=.01)
    solver=GameSolver(model,opt,'optimistic')
    solver.previous=[torch.ones_like(p) for p in solver.parameters]
    ctx=SimpleNamespace(pseudo=torch.tensor([0,0,1,1]),base_mask=torch.ones(4,dtype=torch.bool))
    assert not update_pseudo_problem(solver,ctx,2,.25)
    assert solver.previous is not None
    assert not update_pseudo_problem(solver,ctx,2,.25)
    saved=solver.state_dict()
    restored=GameSolver(model,opt,'optimistic');restored.load_state_dict(saved)
    assert restored.pseudo_signature==solver.pseudo_signature
    ctx.base_mask=torch.zeros(4,dtype=torch.bool)
    assert update_pseudo_problem(solver,ctx,2,.25)
    assert solver.previous is None


def test_rotated_u_blocks_allow_temporal_gate_without_hidden_tx():
    data=SourceData(None,SyntheticRole('unlabeled',1),None,{}, {})
    _,hidden,_,meta=next(iter(data.unlabeled_epoch_loader(12,epoch_index=3,steps=2,seed=392005)))
    assert bool((hidden==-1).all()) and not any('tx' in key for key in meta)
    args=SimpleNamespace(pseudo_temporal_min_conf=.9,pseudo_temporal_window=2)
    accepted=temporal_mask(torch.zeros(12,dtype=torch.long),torch.ones(12),meta,args)
    assert bool(accepted.all())


def test_pseudo_window_retains_history_between_observations():
    model=torch.nn.Linear(1,1);solver=GameSolver(model,torch.optim.SGD(model.parameters(),lr=.1),'optimistic')
    solver.previous=[torch.ones_like(p) for p in solver.parameters]
    ctx=SimpleNamespace(pseudo=torch.tensor([0,0,0,0]),base_mask=torch.ones(4,dtype=torch.bool))
    for _ in range(3):
        assert not update_pseudo_problem(solver,ctx,2,.25,window=4)
        assert solver.previous is not None
    assert solver.pseudo_signature is None and solver.pseudo_accumulator['batches']==3
    state=solver.state_dict();restored=GameSolver(model,solver.optimizer,'optimistic');restored.load_state_dict(state)
    assert not update_pseudo_problem(restored,ctx,2,.25,window=4)
    assert restored.pseudo_signature['selected_fraction']==1
