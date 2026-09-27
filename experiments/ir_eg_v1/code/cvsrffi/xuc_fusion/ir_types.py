"""Explicit contracts for the head-only IR response; no persistent caches."""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import sys
from typing import Protocol
import torch

TensorTuple = tuple[torch.Tensor, ...]
GradientTuple = tuple[torch.Tensor | None, ...]

@dataclass(frozen=True)
class ParamLayout:
    names: tuple[str, ...]
    shapes: tuple[torch.Size, ...]
    phi_indices: tuple[int, ...]
    psi_indices: tuple[int, ...]
    aliases: dict[str, str]
    signature: str

@dataclass(frozen=True)
class DomainCall:
    key: str
    physical_ids: tuple[str, ...]
    z_detached: torch.Tensor
    domain: torch.Tensor
    sample_weight: torch.Tensor
    scaled_dropout_mask: torch.Tensor
    grl_multiplier: float

@dataclass(frozen=True)
class HeadBatch:
    calls: tuple[DomainCall, ...]
    layout_signature: str
    context_signature: str

@dataclass(frozen=True)
class ResponseMetric:
    diagonal: torch.Tensor
    sqrt_diagonal: torch.Tensor
    active_indices: torch.Tensor
    predictor_clip_coefficient: float

@dataclass(frozen=True)
class CGResult:
    solution: torch.Tensor
    status: str
    iterations: int
    relative_residual: float
    reason: str | None

class ObjectivePacket(Protocol):
    head_batch: HeadBatch
    phi_predictor: TensorTuple
    def assemble(self, phi_leaf: TensorTuple) -> torch.Tensor: ...
    def release(self) -> None: ...

def build_layout(model, optimizer) -> ParamLayout:
    if getattr(model, 'adv_head', None) is None:
        raise ValueError('IR requires the native adv_head')
    canonical={};aliases={};objects={}
    for name,p in model.named_parameters(remove_duplicate=False):
        if id(p) in canonical:aliases[name]=canonical[id(p)]
        else:canonical[id(p)]=name;objects[id(p)]=p
    params=[];seen=set()
    for group in optimizer.param_groups:
        for p in group['params']:
            if id(p) not in canonical:raise ValueError('optimizer parameter outside model')
            if id(p) not in seen:params.append(p);seen.add(id(p))
    head_ids={id(p) for p in model.adv_head.parameters()}
    if not head_ids.issubset(seen):raise ValueError('unsupported layout: adv_head forward parameter missing from optimizer, including frozen tensors')
    names=tuple(canonical[id(p)] for p in params);shapes=tuple(p.shape for p in params)
    positions={id(p):i for i,p in enumerate(params)}
    phi=tuple(positions[id(p)] for p in model.adv_head.parameters())
    psi=tuple(i for i,p in enumerate(params) if id(p) not in head_ids)
    payload=(names,[list(s) for s in shapes],phi,psi,sorted(aliases.items()))
    signature=hashlib.sha256(json.dumps(payload,separators=(',',':')).encode()).hexdigest()
    return ParamLayout(names,shapes,phi,psi,aliases,signature)

build_param_layout=build_layout

def assert_import_origins(expected):
    result={}
    for name,path in expected.items():
        module=sys.modules.get(name)
        actual=getattr(module,'__file__',None)
        if actual is None or Path(actual).resolve()!=Path(path).resolve():
            raise RuntimeError(f'import origin mismatch: {name}: {actual} != {path}')
        result[name]={'path':str(Path(actual).resolve()),'sha256':hashlib.sha256(Path(actual).read_bytes()).hexdigest()}
    return result

_RUNTIME_MODULES=(
    'model_dual_cvsincnet', 'cvsrffi.game_tracking.solvers',
    'cvsrffi.game_tracking.state', 'cvsrffi.game_tracking.legacy.objective',
    'cvsrffi.xuc_fusion.objective', 'cvsrffi.xuc_fusion.dr_objective',
    'cvsrffi.xuc_fusion.joint_normalization', 'SSDG.train_ssdg',
)
_OPTIONAL_NATIVE_MODULES=('model','model_modified','cvsrffi.game_tracking.legacy.model',
    'cvsrffi.xuc_fusion.ir_types','cvsrffi.xuc_fusion.ir_head',
    'cvsrffi.xuc_fusion.ir_metric','cvsrffi.xuc_fusion.ir_cg',
    'cvsrffi.xuc_fusion.ir_solver','cvsrffi.xuc_fusion.ir_objective')

def _package_imports(expected_code_root, *, require_runtime):
    """The root is supplied by the launcher, never inferred from this module."""
    root=Path(expected_code_root).resolve(strict=True)
    if not (root/'cvsrffi'/'xuc_fusion').is_dir():raise ValueError('expected_code_root is not an execution package code directory')
    names=set(_RUNTIME_MODULES if require_runtime else ())
    names.update(name for name in _RUNTIME_MODULES+_OPTIONAL_NATIVE_MODULES if name in sys.modules)
    names.add('cvsrffi.xuc_fusion.ir_types')
    return assert_import_origins({name:root.joinpath(*name.split('.')).with_suffix('.py') for name in sorted(names)})

def audit_runtime_imports(expected_code_root):
    """Require critical native runtime imports to come from the launcher's root.

    Call after importing the native objective/solver stack. Math-only contract
    audits check any loaded native modules without importing the runtime stack.
    """
    return _package_imports(expected_code_root,require_runtime=True)

def _reject_target_fields(value, prefix='ctx', seen=None):
    seen=set() if seen is None else seen
    if id(value) in seen:return
    seen.add(id(value))
    if isinstance(value,dict):items=value.items()
    elif hasattr(value,'__dict__') and not isinstance(value,(torch.Tensor,torch.nn.Module)):items=vars(value).items()
    elif isinstance(value,(list,tuple)):
        for i,v in enumerate(value):_reject_target_fields(v,f'{prefix}[{i}]',seen)
        return
    else:return
    for k,v in items:
        key=str(k).lower()
        if 'target' in key or key in {'truth','hidden_tx','u_tx_labels'}:
            raise ValueError(f'forbidden target/truth field: {prefix}.{k}')
        _reject_target_fields(v,f'{prefix}.{k}',seen)

def audit_contract(model, optimizer, ctx) -> dict:
    """Audit real graph ownership, never substitute a coefficient-only ledger.

    ctx supplies head_batch, task_terms (named actual tensors), native_loss,
    and optionally expected_imports. This function retains caller graphs.
    """
    from .ir_head import head_gradient
    _reject_target_fields(ctx)
    data=ctx if isinstance(ctx,dict) else vars(ctx)
    layout=build_layout(model,optimizer)
    required={'head_batch','task_terms','native_loss','expected_code_root'}
    if not required.issubset(data):raise ValueError('contract needs actual native_loss, task_terms, head_batch and independent expected_code_root')
    imports=_package_imports(data['expected_code_root'],require_runtime=False)
    batch=data['head_batch']
    if batch.layout_signature!=layout.signature:raise ValueError('head layout changed')
    params=[];seen=set()
    for group in optimizer.param_groups:
        for p in group['params']:
            if id(p) not in seen:params.append(p);seen.add(id(p))
    phi=tuple(params[i] for i in layout.phi_indices)
    active=tuple(p for p in phi if p.requires_grad)
    for name,term in data['task_terms'].items():
        if not isinstance(term,torch.Tensor):raise ValueError('task ledger must contain actual tensors')
        gs=torch.autograd.grad(term,active,retain_graph=True,allow_unused=True) if term.requires_grad and active else ()
        if any(g is not None and (not torch.isfinite(g).all() or torch.count_nonzero(g)>0) for g in gs):
            raise ValueError(f'task term has adv_head gradient: {name}')
    native=torch.autograd.grad(data['native_loss'],active,retain_graph=True,allow_unused=True) if active else ()
    actual=torch.cat([(torch.zeros_like(p) if g is None else g).reshape(-1) for p,g in zip(active,native)]) if active else phi[0].new_empty(0)
    replay=head_gradient(phi,batch)
    pieces=replay.split([p.numel() for p in phi])
    expected=torch.cat([g for p,g in zip(phi,pieces) if p.requires_grad]) if active else phi[0].new_empty(0)
    if not torch.allclose(actual,expected,atol=1e-6,rtol=1e-5):raise ValueError('native weighted head gradient ownership mismatch')
    imports.update(assert_import_origins(data.get('expected_imports',{})))
    return {'layout_signature':layout.signature,'gradient_ownership_verified':True,
            'imports':imports,
            'task_terms':list(data['task_terms']),
            'domain_calls':[{'key':c.key,'physical_ids':list(c.physical_ids),'sample_weights':c.sample_weight.detach().cpu().tolist(),
                             'effective_coefficient':float(c.sample_weight.sum()),'grl_multiplier':c.grl_multiplier} for c in batch.calls]}
