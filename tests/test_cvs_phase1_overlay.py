from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

from experiments.cvs_phase1_overlay.model import Phase1IdentityAdapter, native_modules


def args(mixstyle=False):
    return SimpleNamespace(num_classes=6, input_len=256, use_mixstyle=mixstyle,
        mixstyle_p=1., mixstyle_layers='time_down,t1',
        mixstyle_mix='same_tx_crossdomain', mixstyle_fallback='skip')


def test_adapter_preserves_selected_network_and_real_branch_features():
    native_modules()
    from experiments.cvs_reference_identity.model import build
    torch.set_num_threads(2)
    torch.manual_seed(2026092701)
    original = build('reference_response').eval()
    torch.manual_seed(2026092701)
    adapter = Phase1IdentityAdapter(args()).eval()
    x = torch.randn(4, 2, 256)
    with torch.no_grad():
        expected = original(x)
        aux = adapter(x, return_aux=True)
    torch.testing.assert_close(aux['logits'], expected, rtol=0, atol=0)
    assert aux['feat_joint'].shape == (4, 160)
    assert aux['feat_imp'].abs().sum() > 0
    assert not torch.equal(aux['feat_imp'], aux['feat_cls'])
    assert sum(p.numel() for p in adapter.parameters()) == 177025


def test_mixstyle_reaches_two_registered_locations_and_ema_is_independent():
    native_modules()
    model = Phase1IdentityAdapter(args(True)).train()
    x = torch.randn(4, 2, 256)
    y = torch.tensor([0, 0, 1, 1])
    d = torch.tensor([0, 1, 0, 1])
    aux = model(x, y=y, domain_labels=d, return_aux=True)
    torch.nn.functional.cross_entropy(aux['logits'], y).backward()
    assert model.last_mixstyle_calls == 2
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    ema = deepcopy(model).eval()
    before = {k: v.clone() for k, v in ema.state_dict().items()}
    with torch.no_grad():
        ema(x, return_aux=True)
    assert ema.last_mixstyle_calls == 0
    for k, v in before.items():
        torch.testing.assert_close(ema.state_dict()[k], v, rtol=0, atol=0)


def test_unsupported_interface_rejected():
    a = args()
    a.num_classes = 5
    with pytest.raises(ValueError):
        Phase1IdentityAdapter(a)


def test_epoch80_satellite_ce_and_early_concat_are_distinct():
    from experiments.cvs_phase1_overlay.augmentation import loss_for_batch
    class Tiny(torch.nn.Module):
        def forward(self,x,**kwargs):return x
    x=torch.tensor([[2.,0.],[0.,2.]],requires_grad=True);y=torch.tensor([0,1]);d=torch.tensor([0,1])
    aug=lambda *a:SimpleNamespace(x=-x,scenario='fake',view_prob=1.,applied=True)
    early,e=loss_for_batch(Tiny(),x,y,d,{},aug,79,1)
    late,l=loss_for_batch(Tiny(),x,y,d,{},aug,80,1)
    assert e['concat_samples']==2 and e['satellite_ce'] is None
    torch.testing.assert_close(early,torch.nn.functional.cross_entropy(x,y))
    torch.testing.assert_close(late,early+.68*torch.nn.functional.cross_entropy(-x,y))


def test_frozen_matrix_rejects_missing_duplicate_and_wrong_paths():
    from experiments.cvs_phase1_overlay.contract import ROOT,read
    from experiments.cvs_phase1_overlay.dispatch import validate_matrix
    spec=read(ROOT/'experiments/cvs_phase1_overlay/configs/launch_spec.json');validate_matrix(spec)
    for change in ['missing','duplicate','path']:
        c=deepcopy(spec)
        if change=='missing':c['rows'].pop()
        elif change=='duplicate':c['rows'][0]=c['rows'][1]
        else:c['rows'][0]['source_output']+='/../other'
        with pytest.raises(ValueError):validate_matrix(c)


def test_source_rejects_target_and_checkpoint_and_recipe_drift():
    from experiments.cvs_phase1_overlay.contract import ROOT,read,validate_config
    c=read(ROOT/'experiments/cvs_phase1_overlay/configs/source-ce-s2026092701.json');validate_config(c)
    for key,value in [('checkpoint','old.pt'),('p1_capsule','query'),('epochs',100),('target_access',True)]:
        bad=deepcopy(c);bad[key]=value
        with pytest.raises(ValueError):validate_config(bad)


def test_mixstyle_skips_ineligible_pairs_and_evaluation():
    m=Phase1IdentityAdapter(args(True)).train();x=torch.randn(4,2,256);y=torch.arange(4);d=torch.arange(4)
    m(x,y=y,domain_labels=d)
    assert m.last_mixstyle_changed_samples==0
    m.eval();m(x,y=y,domain_labels=d)
    assert m.last_mixstyle_calls==0


def test_independent_recount_and_interaction():
    import numpy as np
    from experiments.cvs_phase1_overlay.analyze import recount_metrics,contrasts
    cm,acc,f1=recount_metrics(np.array([0,0,1,1]),np.array([0,1,1,1]))
    assert cm.sum()==4 and acc==.75
    assert f1==pytest.approx(((2/3)+.8)/6)
    assert contrasts(dict(ce=70,leo=72,mixstyle=73,leo_mixstyle=76))['interaction']==1
