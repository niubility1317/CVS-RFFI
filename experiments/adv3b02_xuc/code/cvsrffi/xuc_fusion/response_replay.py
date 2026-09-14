"""Replay fixed-background leaf computations while recomputing local dependents.

Only deterministic, buffer-free builtins are cached. Autograd dependence alone
does not prove forward independence (detach is a counterexample), so cache hits
also require identical inputs and unchanged parameter versions.
"""
from contextlib import contextmanager
import torch

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

def same(left,right):
    if torch.is_tensor(left):return torch.is_tensor(right) and left.shape==right.shape and left.dtype==right.dtype and left.device==right.device and torch.equal(left,right)
    if type(left) is not type(right):return False
    if isinstance(left,dict):return left.keys()==right.keys() and all(same(left[k],right[k]) for k in left)
    if isinstance(left,(tuple,list)):return len(left)==len(right) and all(same(a,b) for a,b in zip(left,right))
    return left==right

PURE_TYPES=(torch.nn.Linear,torch.nn.Conv1d,torch.nn.Conv2d,torch.nn.ReLU,torch.nn.Tanh,
    torch.nn.SiLU,torch.nn.GELU,torch.nn.ELU,torch.nn.LeakyReLU,torch.nn.Identity,
    torch.nn.Flatten,torch.nn.AdaptiveAvgPool1d,torch.nn.AvgPool1d,torch.nn.MaxPool1d,
    torch.nn.LayerNorm,torch.nn.GroupNorm)

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
                if type(module) not in PURE_TYPES or list(module.children()) or list(module.buffers()):continue
                if any(id(p) in self.selected for p in module.parameters()):continue
                original=module.forward
                had_override='forward' in module.__dict__;old_override=module.__dict__.get('forward')
                def forward(*args,_name=name,_module=module,_original=original,**kwargs):
                    index=counts.get(_name,0);counts[_name]=index+1
                    key=(tag,_name,index,_module.training)
                    versions=tuple(p._version for p in _module.parameters())
                    inputs=(args,kwargs)
                    dependent=self.depends(inputs)
                    if key in self.cache and not dependent:
                        value,old_inputs,old_versions=self.cache[key]
                        if versions==old_versions and same(inputs,old_inputs):
                            self.replayed+=1
                            return detached(value)
                    result=_original(*args,**kwargs);self.executed+=1
                    # Do not learn caches under no_grad: it hides dependency.
                    if torch.is_grad_enabled() and not dependent and not self.depends(result):
                        self.cache[key]=(detached(result),detached(inputs),versions)
                    return result
                saved.append((module,had_override,old_override));module.forward=forward
            yield
        finally:
            for module,had_override,old_override in reversed(saved):
                if had_override:module.forward=old_override
                else:del module.forward
