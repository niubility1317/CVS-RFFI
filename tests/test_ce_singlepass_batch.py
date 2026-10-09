import importlib.util
from pathlib import Path
import pytest

def design():
    path=Path(__file__).resolve().parents[1]/'experiments/cvs_ce_batch_sweep/design.py'
    spec=importlib.util.spec_from_file_location('batchdesign',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module

def test_expected_singlepass_budgets_and_rejection():
    d=design()
    assert [200*__import__('math').ceil(6300/b) for b in d.BATCHES]==[5000,2600,1400,800]
    for n in [5000,2600,1400,800]:d.require_budget(n,n)
    for counts in [(44400,44400),(5000,4999),(25,25)]:
        with pytest.raises(ValueError):d.require_budget(*counts)

def test_normal_pass_covers_each_sample_once_including_tail():
    from torch.utils.data import DataLoader,TensorDataset
    import torch
    for b in [256,512,1024,2048]:
        batches=list(DataLoader(TensorDataset(torch.arange(6300)),batch_size=b,shuffle=True,drop_last=False))
        values=torch.cat([x[0] for x in batches])
        assert len(values)==6300 and len(set(values.tolist()))==6300
        assert len(batches)==__import__('math').ceil(6300/b)

def test_remote_payload_and_nested_preflight_compile():
    import ast
    from experiments.cvs_ce_batch_sweep import publish
    payload=publish.REMOTE.replace('CONFIG','{}')
    tree=ast.parse(payload)
    node=next(n for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='check' for t in n.targets))
    command=eval(compile(ast.Expression(node.value),'<preflight>','eval'),{'release':Path('/tmp/fixed-release')})
    compile(command,'<actual-loader-preflight>','exec')
