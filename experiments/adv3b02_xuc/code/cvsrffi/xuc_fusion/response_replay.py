"""Replay fixed-background leaf computations while recomputing local dependents.

Only lives inside one local game. Parameters outside the selected local blocks
must stay fixed. Dependence is established from autograd, not module name guesses.
"""
from contextlib import contextmanager
import torch
from cvsrffi.game_tracking.state import RNGState

def tensors(value):
    if torch.is_tensor(value):return [value]
    if isinstance(value,dict):return [t for v in value.values() for t in tensors(v)]
    if isinstance(value,(tuple,list)):return [t for v in value for t in tensors(v)]
    return []

def detached(value):
    if torch.is_tensor(value):return value.detach().clone()
    if isinstance(value,dict):return {k:detached(v) for k,v in value.items()}
    if isinstance(value,tuple):return tuple(detached(v) for v in value)
    if isinstance(value,list):return [detached(v) for v in value]
    return value

class LocalReplay:
    def __init__(self,model,selected):
        self.model=model;self.selected={id(p) for p in selected};self.cache={}
        self.executed=0;self.replayed=0
    def depends(self,value):
        leaves=tensors(value)
        if any(id(t) in self.selected for t in leaves):return True
        stack=[t.grad_fn for t in leaves if t.grad_fn is not None];seen=set()
        while stack:
            node=stack.pop()
            if node in seen:continue
            seen.add(node)
            if id(getattr(node,'variable',None)) in self.selected:return True
            stack.extend(parent for parent,_ in node.next_functions if parent is not None)
        return False
    @contextmanager
    def session(self,tag):
        saved=[];counts={}
        try:
            for name,module in self.model.named_modules():
                if list(module.children()):continue
                original=module.forward
                def forward(*args,_name=name,_module=module,_original=original,**kwargs):
                    index=counts.get(_name,0);counts[_name]=index+1
                    key=(tag,_name,index,_module.training)
                    if key in self.cache:
                        value,rng=self.cache[key];rng.restore();self.replayed+=1
                        return detached(value)
                    result=_original(*args,**kwargs);self.executed+=1
                    # Do not learn caches under no_grad: it hides dependency.
                    if torch.is_grad_enabled() and not self.depends(result):
                        self.cache[key]=(detached(result),RNGState.capture())
                    return result
                saved.append((module,original));module.forward=forward
            yield
        finally:
            for module,original in reversed(saved):module.forward=original
