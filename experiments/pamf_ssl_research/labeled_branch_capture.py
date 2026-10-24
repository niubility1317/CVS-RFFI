"""Step-local reuse of the four native L pairs;no additional model forward."""
from contextlib import contextmanager
from contextvars import ContextVar
import torch

_CURRENT=ContextVar('pamf_training_L_branch_pairs',default=None)


def labeled_context_active():return _CURRENT.get() is not None


@contextmanager
def capture():
    from experiments.cric_dura_comparison import t1_model
    if labeled_context_active():raise ValueError('One native labeled branch context at a time')
    pairs=[];token=_CURRENT.set(pairs);original=t1_model.pair
    def pair(*args,**kwargs):
        value=original(*args,**kwargs);pairs.append(value);return value
    t1_model.pair=pair
    try:yield
    finally:
        t1_model.pair=original;_CURRENT.reset(token);pairs.clear()


def components(labeled):
    pairs=_CURRENT.get()
    if pairs is None or len(pairs)!=4:raise ValueError('Exactly four original L pairs in the current labeled hook required')
    clean=.5*pairs[0][0]+.25*pairs[1][0]+.25*pairs[3][0]
    physical=.25*pairs[2][0]
    if not bool(torch.isfinite(clean+physical+labeled)):raise ValueError('Finite complete and component L losses required')
    return clean,physical


@contextmanager
def count_autograd_calls():
    calls=[];original=torch.autograd.grad
    def counted(*args,**kwargs):
        calls.append(1);return original(*args,**kwargs)
    torch.autograd.grad=counted
    try:yield calls
    finally:torch.autograd.grad=original
