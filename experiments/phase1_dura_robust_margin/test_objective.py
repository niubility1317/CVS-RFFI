import unittest
import torch
from torch import nn
from experiments.phase1_dura_robust_margin.objective import adversarial_reader_loss, robust_source_objective
from experiments.phase1_dura_context.model import ContextRecovery


class Reader(nn.Module):
    def __init__(self):
        super().__init__()
        self.head = nn.Linear(512, 6)
        self.requires_grad_(False)
    def features(self, x):
        return x.reshape(len(x), -1)
    def classify_features(self, f):
        return self.head(f)


class RobustObjectiveTest(unittest.TestCase):
    def test_bounded_per_record_attack_and_trainable_output(self):
        torch.manual_seed(56); reader=Reader().eval()
        output=torch.randn(4,2,256,requires_grad=True)
        clean_features=reader.features(torch.randn(4,2,256)).detach()
        groups=torch.tensor([4,4,6,6])
        loss,delta=adversarial_reader_loss(reader,output,clean_features,groups,epsilon=0.01)
        self.assertFalse(delta.requires_grad)
        attack_rms=delta.square().mean((1,2)).sqrt()
        signal_rms=output.detach().square().mean((1,2)).sqrt()
        self.assertTrue(torch.all(attack_rms <= 0.010001*signal_rms))
        self.assertTrue(torch.all(attack_rms > 0))
        loss.backward()
        self.assertTrue(torch.isfinite(output.grad).all())
        self.assertGreater(output.grad.abs().sum().item(),0)
        self.assertTrue(all(p.grad is None for p in reader.parameters()))

    def test_full_protected_objective_trains_zero_initialized_model(self):
        torch.manual_seed(91); torch.set_num_threads(2)
        reader=Reader().eval(); model=ContextRecovery()
        x=torch.randn(4,2,256);anchor=x+0.05*torch.randn_like(x);clean=torch.randn_like(x)
        target_features=reader.features(clean).detach();groups=torch.tensor([4,4,6,6])
        output=model(x,anchor,torch.full((4,),-2.0))
        loss,terms=robust_source_objective(reader,output,anchor,clean,target_features,groups)
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(terms['reader_stability']))
        loss.backward()
        self.assertTrue(torch.isfinite(model.gate_output.weight.grad).all())
        self.assertGreater(model.gate_output.weight.grad.abs().sum().item(),0)
        self.assertTrue(all(p.grad is None for p in reader.parameters()))

    def test_rejects_bad_groups_and_no_attack_training_gradient(self):
        reader=Reader().eval();output=torch.randn(3,2,256,requires_grad=True)
        clean_features=reader.features(torch.randn(3,2,256)).detach()
        with self.assertRaises(ValueError):adversarial_reader_loss(reader,output,clean_features,torch.tensor([4,6]),epsilon=0.01)
        with self.assertRaises(ValueError):adversarial_reader_loss(reader,output.detach(),clean_features,torch.tensor([4,4,6]),epsilon=0.01)

if __name__=='__main__':unittest.main()
