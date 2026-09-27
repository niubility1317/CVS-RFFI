import os
import pytest

@pytest.fixture(autouse=True)
def _acceptance_device():
    device=os.environ.get('IR_ACCEPTANCE_DEVICE','cpu')
    if device not in ('cpu','cuda'):raise ValueError('invalid IR_ACCEPTANCE_DEVICE')
    with torch.device(device):
        yield

from types import SimpleNamespace
from pathlib import Path
import pytest
import torch
from torch import nn
from cvsrffi.xuc_fusion.ir_types import build_layout, audit_contract, assert_import_origins, audit_runtime_imports

CODE_ROOT=Path(__file__).resolve().parents[1]/'code'

def model():
    m=nn.Module();m.body=nn.Linear(2,2);m.dom_head=nn.Linear(2,2)
    m.adv_head=nn.Sequential(nn.Linear(2,3),nn.ReLU(),nn.Dropout(.2),nn.Linear(3,2))
    return m

def test_adv_head_not_dom_head():
    m=model();layout=build_layout(m,torch.optim.AdamW(m.parameters()))
    assert all(layout.names[i].startswith('adv_head.') for i in layout.phi_indices)
    assert any(layout.names[i].startswith('dom_head.') for i in layout.psi_indices)

def test_shared_parameter_dedup():
    m=model();m.alias=m.adv_head
    layout=build_layout(m,torch.optim.AdamW(m.parameters()))
    assert len(layout.names)==len(list(m.parameters())) and layout.aliases['alias.0.weight']=='adv_head.0.weight'

def test_head_order_survives_optimizer_group_reordering():
    m=model();head=list(m.adv_head.parameters());other=list(m.body.parameters())+list(m.dom_head.parameters())
    opt=torch.optim.AdamW([{'params':head[::-1]},{'params':other}]);layout=build_layout(m,opt)
    assert tuple(layout.names[i] for i in layout.phi_indices)==tuple('adv_head.'+n for n,_ in m.adv_head.named_parameters())

def test_import_origin_is_pinned(tmp_path):
    import cvsrffi.xuc_fusion.ir_types as mod
    assert_import_origins({'cvsrffi.xuc_fusion.ir_types':mod.__file__})
    with pytest.raises(RuntimeError,match='import'):assert_import_origins({'cvsrffi.xuc_fusion.ir_types':str(tmp_path/'wrong.py')})

def test_no_target_fields_in_context():
    m=model();opt=torch.optim.AdamW(m.parameters())
    with pytest.raises(ValueError,match='target'):audit_contract(m,opt,{'nested':{'target_truth':torch.tensor([1])}})

def _audit(extra=False,frozen=False,expected_code_root=CODE_ROOT):
    from cvsrffi.xuc_fusion.ir_head import capture_head_calls
    m=model()
    if frozen:m.adv_head[0].bias.requires_grad_(False)
    opt=torch.optim.AdamW(m.parameters());layout=build_layout(m,opt)
    with capture_head_calls(m.adv_head,layout.signature,'ctx') as tape:
        z=m.body(torch.randn(2,2)); a=m.adv_head(z); b=m.adv_head(torch.randn(5,2))
        for i,(n,w) in enumerate(((2,.7),(5,.2))):
            tape.register(str(i),tuple(map(str,range(n))),torch.zeros(n,dtype=torch.long),torch.full((n,),w/n),1. if i==0 else 0.,call_index=i)
    task=z.square().sum()+(a.sum()*.01 if extra else 0.)
    native=task+.7*torch.nn.functional.cross_entropy(a,torch.zeros(2,dtype=torch.long))+.2*torch.nn.functional.cross_entropy(b,torch.zeros(5,dtype=torch.long))
    return audit_contract(m,opt,{'head_batch':tape.batch(),'task_terms':{'identity':task},'native_loss':native,'expected_code_root':expected_code_root})

def test_task_has_no_phi_gradient():
    with pytest.raises(ValueError,match='task'): _audit(True)

def test_weighted_LU_reduction_matches_native():
    report=_audit();assert report['gradient_ownership_verified'] and len(report['domain_calls'])==2

def test_frozen_head_and_nonhead_slots_preserved_under_group_reorder():
    m=model();m.adv_head[0].bias.requires_grad_(False);m.body.weight.requires_grad_(False)
    head=list(m.adv_head.parameters());other=list(m.body.parameters())+list(m.dom_head.parameters())
    opt=torch.optim.AdamW([{'params':head[::-1]},{'params':other}]);layout=build_layout(m,opt)
    assert len(layout.phi_indices)==4
    assert set(layout.phi_indices+layout.psi_indices)==set(range(len(layout.names)))
    assert tuple(layout.names[i] for i in layout.phi_indices)==tuple('adv_head.'+n for n,_ in m.adv_head.named_parameters())
    assert _audit(frozen=True)['gradient_ownership_verified']

def test_missing_frozen_head_optimizer_slot_is_explicitly_unsupported():
    m=model();omitted=m.adv_head[0].bias;omitted.requires_grad_(False)
    opt=torch.optim.AdamW([p for p in m.parameters() if p is not omitted])
    with pytest.raises(ValueError,match='forward parameter missing from optimizer'):build_layout(m,opt)

def test_contract_rejects_foreign_native_module_without_optional_override(monkeypatch,tmp_path):
    import sys
    monkeypatch.setitem(sys.modules,'model_dual_cvsincnet',SimpleNamespace(__file__=str(tmp_path/'model_dual_cvsincnet.py')))
    with pytest.raises(RuntimeError,match='model_dual_cvsincnet'):_audit()

def test_contract_root_is_independent_of_loaded_helper(tmp_path):
    (tmp_path/'cvsrffi'/'xuc_fusion').mkdir(parents=True)
    with pytest.raises(RuntimeError,match='import origin mismatch'):_audit(expected_code_root=tmp_path)

def test_runtime_import_audit_requires_critical_modules_and_rejects_foreign(monkeypatch,tmp_path):
    import sys
    from cvsrffi.xuc_fusion.ir_types import _RUNTIME_MODULES
    for name in _RUNTIME_MODULES:
        path=CODE_ROOT.joinpath(*name.split('.')).with_suffix('.py')
        monkeypatch.setitem(sys.modules,name,SimpleNamespace(__file__=str(path)))
    assert 'SSDG.train_ssdg' in audit_runtime_imports(CODE_ROOT)
    monkeypatch.setitem(sys.modules,'model_dual_cvsincnet',SimpleNamespace(__file__=str(tmp_path/'model_dual_cvsincnet.py')))
    with pytest.raises(RuntimeError,match='model_dual_cvsincnet'):audit_runtime_imports(CODE_ROOT)
