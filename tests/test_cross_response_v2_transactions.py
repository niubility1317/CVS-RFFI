import copy
import torch
import pytest

from cvsrffi.cross_response.training import IndependentAuxiliaryTransaction
from cvsrffi.cross_response.cache import FixedStatisticCache


def test_legacy_torch_without_unified_amp_scaler(monkeypatch):
    monkeypatch.delattr(torch.amp, 'GradScaler')
    head = torch.nn.Linear(2, 1)
    transaction = IndependentAuxiliaryTransaction(head.parameters(), lr=.01, weight_decay=0., amp=False)
    before = head.weight.detach().clone()
    result = transaction.step(head(torch.ones(2, 2)).square().mean())
    assert result['auxiliary_step_applied'] == 1
    assert not torch.equal(before, head.weight)


@pytest.mark.parametrize("bad", [None, float("nan"), float("inf")])
def test_head_only_main_update_isolated(bad):
    torch.manual_seed(39)
    a = torch.nn.Linear(3, 2)
    b = copy.deepcopy(a)
    head = torch.nn.Linear(2, 1)
    oa = torch.optim.AdamW(a.parameters(), lr=.01)
    ob = torch.optim.AdamW(b.parameters(), lr=.01)
    aux = IndependentAuxiliaryTransaction(head.parameters(), lr=.1, weight_decay=.1, max_grad_norm=.00001)
    x = torch.randn(4, 3)
    for step in range(3):
        oa.zero_grad(set_to_none=True)
        ob.zero_grad(set_to_none=True)
        za, zb = a(x), b(x)
        la, lb = za.square().mean(), zb.square().mean()
        response = head(zb).square().mean()
        if bad is not None:
            response = response * bad
        log = aux.step(response)
        assert bool(log["auxiliary_step_applied"]) == (bad is None)
        la.backward()
        lb.backward()
        for pa, pb in zip(a.parameters(), b.parameters()):
            torch.testing.assert_close(pa.grad, pb.grad, rtol=0, atol=0)
        torch.nn.utils.clip_grad_norm_(a.parameters(), .2)
        torch.nn.utils.clip_grad_norm_(b.parameters(), .2)
        oa.step()
        ob.step()
        for pa, pb in zip(a.parameters(), b.parameters()):
            torch.testing.assert_close(pa, pb, rtol=0, atol=0)
        for sa, sb in zip(oa.state.values(), ob.state.values()):
            for k in sa:
                torch.testing.assert_close(sa[k], sb[k], rtol=0, atol=0)


def test_cache_keys_and_detachment():
    cache = FixedStatisticCache()
    common = dict(crop=(0, 256), statistics_config={"family": "fft", "bands": 8})
    key = cache.key("physical1", **common)
    expected = torch.arange(8).float()[None]
    cache.get_many([key], lambda _: expected, device="cpu")
    got = cache.get_many([key], lambda _: pytest.fail("must hit"), device="cpu")
    torch.testing.assert_close(got, expected, rtol=0, atol=0)
    assert cache.report()["value_bytes"] == 32
    assert cache.key("physical1", **common, scale_version="source_scale_v2") != key
    assert cache.key("physical1", crop=(1,257), statistics_config=common["statistics_config"]) != key
    with pytest.raises(ValueError):
        cache.get_many([("learned",)], lambda _: torch.ones(1,8,requires_grad=True), device="cpu")


def test_decomposed_backward_routes_once_to_exact_parameter_groups():
    from cvsrffi.cross_response.training import response_backward
    aux, identity, domain, shared = [torch.nn.Parameter(torch.tensor(1.)) for _ in range(4)]
    roles = {"auxiliary":[("aux",aux)],"identity_tail":[("id",identity)],
             "domain":[("dom",domain)],"shared":[("shared",shared)],"identity_front":[],"other":[]}
    full = 2*aux + 100*identity + 100*domain + 100*shared
    identity_route = 3*identity + 100*aux + 100*domain + 100*shared
    domain_route = 5*domain + 100*aux + 100*identity + 100*shared
    baseline = identity.square() + domain.square() + shared.square()
    response_backward(baseline_loss=baseline,identity_loss=identity.square(),response_loss=full,
        decision_loss=baseline*0,cross_loss=baseline*0,roles=roles,
        scaler=torch.amp.GradScaler("cuda",enabled=False),lambda_resp=.4,lambda_dec=0,lambda_cross=0,
        joint_open=True,gradient_cap=1000, response_group_losses={
            "predictor_full":full,"identity":identity_route,"domain":domain_route})
    for parameter,expected in ((aux,.8),(identity,3.2),(domain,4.),(shared,2.)):
        torch.testing.assert_close(parameter.grad,torch.tensor(expected))
