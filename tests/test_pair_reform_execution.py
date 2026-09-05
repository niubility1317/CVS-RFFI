import torch
from SSDG.train_ssdg import _compute_pair_unlabeled_step, _backward_with_daot_persistent_projection
from cvsrffi.daot_gradient_control import PersistentConflictProjector
from test_pair_reform_runtime import Toy
from test_pair_reform_training import args_for


def test_warmup_u_has_no_augmentation_or_teacher():
    model, teacher = Toy(), Toy()
    args = args_for('point')
    args.pair_start_epoch = args.pair_pseudo_start_epoch = 11
    def forbidden(*a, **k):
        raise AssertionError('disabled augmentation executed')
    result, _, _, mask = _compute_pair_unlabeled_step(model=model,ema_model=teacher,
        x_u=torch.randn(4,4),d_u=None,args=args,epoch=1,physical_ids=[1,2,3,4],apply_sat_fn=forbidden)
    assert model.calls == [4] and not teacher.calls
    assert result['weighted_components'] == {}
    assert not mask.any()


def test_projection_reductions_are_independent_of_amp_scale():
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.id_backbone=torch.nn.Linear(2,1,bias=False)
            self.id_backbone.weight.data.fill_(1.)
    class Scaler:
        def __init__(self, scale): self.value=scale
        def scale(self, loss): return loss*self.value
        def get_scale(self): return self.value
    results=[]
    for scale in [1., 2.**80]:
        m=Model(); w=m.id_backbone.weight
        info=_backward_with_daot_persistent_projection(m,Scaler(scale),base_loss=w.sum(),
            auxiliary_groups={'orbit':-w.sum()},controller=PersistentConflictProjector(window=1))
        results.append(w.grad/scale)
        assert info['orbit']['projected'] and abs(info['orbit']['cosine']+1)<1e-6
    assert torch.allclose(results[0],results[1])
