import pytest
import torch

from cvsrffi.ecrs_v2 import schur_ridge


@pytest.mark.parametrize('diagnostics', [False, True])
def test_schur_accepts_torch21_single_dimension_boolean_reductions(monkeypatch, diagnostics):
    torch.manual_seed(72)
    y, n, p = torch.randn(2, 32), torch.randn(2, 32, 3), torch.randn(2, 32, 8)
    expected = schur_ridge(y, n, p, diagnostics=diagnostics)
    native_all = torch.Tensor.all
    def torch21_all(tensor, *args, **kwargs):
        dim = kwargs.get('dim', args[0] if args else None)
        if isinstance(dim, (tuple, list)):
            raise TypeError('PyTorch 2.1 Tensor.all supports a single dimension only')
        return native_all(tensor, *args, **kwargs)
    monkeypatch.setattr(torch.Tensor, 'all', torch21_all)
    actual = schur_ridge(y, n, p, diagnostics=diagnostics)
    torch.testing.assert_close(actual['theta'], expected['theta'], atol=0, rtol=0)
    torch.testing.assert_close(actual['eta'], expected['eta'], atol=0, rtol=0)
