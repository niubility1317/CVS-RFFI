import unittest
import torch
from experiments.phase1_dura_feedback.model import pack_context
from experiments.phase1_dura_recurrent_feedback.model import RecurrentFeedbackRecovery


class RecurrentTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(59)
        self.x = torch.randn(3, 2, 256)
        self.anchor = torch.randn(3, 2, 256)
        self.packed = pack_context(self.anchor, torch.randn(3, 160), torch.randn(3, 6))
        self.lv = torch.randn(3)

    def test_zero_initialization_returns_anchor(self):
        model = RecurrentFeedbackRecovery()
        torch.testing.assert_close(model(self.x, self.packed, self.lv), self.anchor, rtol=0, atol=0)

    def test_exactly_three_shared_steps_and_gradients(self):
        model = RecurrentFeedbackRecovery()
        calls, conditions = [], []
        def capture(module, args, out):
            out.retain_grad()
            calls.append(out)
            conditions.append(args[1][:, 2].detach().clone())
        handle = model.step.register_forward_hook(capture)
        out = model(self.x, self.packed, self.lv)
        handle.remove()
        self.assertEqual(len(calls), 3)
        for condition in conditions:
            torch.testing.assert_close(condition, self.packed[:, 2], rtol=0, atol=0)
        out.square().mean().backward()
        self.assertGreater(float(model.gate_output.weight.grad.abs().sum()), 0)
        self.assertGreater(float(calls[0].grad.abs().sum()), 0)
        self.assertTrue(torch.isfinite(model.gate_output.weight.grad).all())

    def test_single_window_inference_is_independent(self):
        model = RecurrentFeedbackRecovery().eval()
        with torch.no_grad():
            model.gate_output.weight.normal_(0, 0.01)
            all_rows = model(self.x, self.packed, self.lv)
            one = model(self.x[:1], self.packed[:1], self.lv[:1])
        torch.testing.assert_close(one, all_rows[:1], atol=3e-5, rtol=3e-5)

    def test_bad_input_is_rejected(self):
        with self.assertRaises(ValueError):
            RecurrentFeedbackRecovery()(self.x, self.packed[:, :2], self.lv)


if __name__ == '__main__':
    unittest.main()
