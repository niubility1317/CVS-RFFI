import importlib
import unittest
import torch
from torch import nn
from experiments.phase1_dura_blendgate.model import AlignedBlend, align_to_anchor


class FakeReader(nn.Module):
    def __init__(self):
        super().__init__()
        self.head = nn.Linear(512, 6)
        self.eval().requires_grad_(False)
    def features(self, x):
        return x.flatten(1)
    def classify_features(self, x):
        return self.head(x)


class FakeExpert(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))
    def forward(self, received, initial, log_variance):
        return self.scale * torch.roll(received, 3, -1)


def rotate(x, angle):
    c, s = torch.cos(angle), torch.sin(angle)
    return torch.stack((c*x[:,0]-s*x[:,1], s*x[:,0]+c*x[:,1]), dim=1)


class ComplexBlendTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(71); torch.set_num_threads(2)
        self.x = torch.randn(3,2,256)
        self.anchor = self.x + .2*torch.randn_like(self.x)
        self.lv = torch.full((3,), -2.0)

    def model(self, template=None):
        try:
            mod = importlib.import_module('experiments.phase1_dura_complexblend.model')
        except ModuleNotFoundError as exc:
            if exc.name == 'experiments.phase1_dura_complexblend.model':
                self.fail('complex edge fusion is not implemented')
            raise
        return mod.ComplexAlignedBlend(template or AlignedBlend(FakeExpert(),preserve_edges=True))

    def test_initial_model_returns_anchor_and_freezes_expert(self):
        model = self.model()
        self.assertTrue(torch.equal(model(self.x,self.anchor,self.lv),self.anchor))
        self.assertTrue(all(not p.requires_grad for p in model.expert.parameters()))

    def test_real_only_gate_matches_existing_edge_fusion(self):
        template = AlignedBlend(FakeExpert(),preserve_edges=True).eval()
        with torch.no_grad():
            template.gate_output.weight.normal_(0,.03)
            template.gate_output.bias.fill_(.2)
        model = self.model(template).eval()
        with torch.no_grad():
            model.gate_output.weight[0].copy_(template.gate_output.weight[0])
            model.gate_output.bias[0].copy_(template.gate_output.bias[0])
        torch.testing.assert_close(model(self.x,self.anchor,self.lv),template(self.x,self.anchor,self.lv),rtol=2e-5,atol=2e-5)

    def test_imaginary_gate_rotates_expert_difference(self):
        model = self.model().eval()
        with torch.no_grad():
            model.gate_output.bias[1] = .4
            y = model(self.x,self.anchor,self.lv)
            expert = model.expert(self.x,self.x,self.lv)
            diff = align_to_anchor(expert,self.anchor,preserve_edges=True)-self.anchor
            expected = self.anchor + .5*torch.tanh(torch.tensor(.4))*torch.stack((-diff[:,1],diff[:,0]),dim=1)
        torch.testing.assert_close(y,expected,rtol=2e-5,atol=2e-5)

    def test_protected_source_loss_trains_gate_without_expert_or_reader_updates(self):
        from experiments.phase1_dura_safe_memory.objective import protected_source_objective
        reader = FakeReader().eval()
        model = self.model()
        clean = self.x + .05 * torch.randn_like(self.x)
        features = reader.features(clean).detach()
        output = model(self.x,self.anchor,self.lv)
        loss, _ = protected_source_objective(reader, output, self.anchor, clean, features, torch.tensor([4,4,6]))
        loss.backward()
        self.assertTrue(torch.isfinite(model.gate_output.weight.grad).all())
        self.assertGreater(float(model.gate_output.weight.grad.abs().sum()), 0)
        self.assertTrue(all(p.grad is None for p in model.expert.parameters()))
        self.assertTrue(all(p.grad is None for p in reader.parameters()))

    def test_nonzero_gate_is_phase_equivariant_and_batch_independent(self):
        model = self.model().eval()
        with torch.no_grad():
            model.gate_output.weight.normal_(0,.02)
            y=model(self.x,self.anchor,self.lv)
            rotated=model(rotate(self.x,torch.tensor(.7)),rotate(self.anchor,torch.tensor(.7)),self.lv)
            first=model(self.x[:1],self.anchor[:1],self.lv[:1])
        torch.testing.assert_close(rotated,rotate(y,torch.tensor(.7)),rtol=3e-5,atol=3e-5)
        torch.testing.assert_close(first,y[:1],rtol=3e-5,atol=3e-5)


if __name__ == '__main__': unittest.main()
