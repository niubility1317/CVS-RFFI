from types import SimpleNamespace
import pytest
import experiments.pamf_dura_efficiency.training_interpolation as m


def setup_fake(monkeypatch):
    def native(seed):return SimpleNamespace(seed=seed,initialization_report={},execution='native')
    base=SimpleNamespace(scratch_model=native)
    def install(model):return lambda mode:setattr(model,'execution',mode)
    monkeypatch.setattr(m,'install_switch',install)
    return base,native


def test_only_training_factory_changes_and_return_restores_native(monkeypatch):
    base,native=setup_fake(monkeypatch)
    def trainer(args,paths):
        g=base.scratch_model(11)
        assert g.seed==11 and g.execution=='slice_linear2x' and 'training_execution' in g.initialization_report
        return args+paths
    assert m.train_with_slices(base,trainer,2,3)==5
    assert base.scratch_model is native and base.scratch_model(11).execution=='native'


def test_training_failure_restores_native_and_propagates(monkeypatch):
    base,native=setup_fake(monkeypatch)
    def trainer(*args):
        base.scratch_model(11)
        raise RuntimeError('scientific run must stop')
    with pytest.raises(RuntimeError,match='scientific run must stop'):m.train_with_slices(base,trainer,None,None)
    assert base.scratch_model is native


def test_install_failure_does_not_leak_factory(monkeypatch):
    base,native=setup_fake(monkeypatch)
    def install(model):raise ValueError('unexpected Core graph')
    monkeypatch.setattr(m,'install_switch',install)
    with pytest.raises(ValueError,match='unexpected Core graph'):
        m.train_with_slices(base,lambda *_:base.scratch_model(11),None,None)
    assert base.scratch_model is native
