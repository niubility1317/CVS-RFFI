import unittest

import torch

from experiments.pamf_lora_main.shared_teacher_direct_run import exact_forward_surrogate, teacher_kl


class TeacherDirectTest(unittest.TestCase):
    def test_exact_forward_and_surrogate_gradient(self):
        proxy = torch.tensor([[2., 5.]], requires_grad=True)
        exact = torch.tensor([[7., 11.]])
        out = exact_forward_surrogate(exact, proxy)
        torch.testing.assert_close(out, exact, rtol=0, atol=0)
        out.sum().backward()
        torch.testing.assert_close(proxy.grad, torch.ones_like(proxy))

    def test_kl_vanishes_for_same_distribution_and_gradients_reach_student(self):
        target = torch.tensor([[2., 0.], [0., 2.]])
        student = target.clone().requires_grad_(True)
        self.assertLess(abs(teacher_kl(target, student).item()), 1e-6)
        teacher_kl(target, student+torch.tensor([[.2, 0.], [.1, 0.]])).backward()
        self.assertTrue(torch.isfinite(student.grad).all())
        self.assertGreater(student.grad.abs().sum().item(), 0)

    def test_rejects_mismatched_or_nonfinite_frontend(self):
        with self.assertRaises(ValueError):
            exact_forward_surrogate(torch.zeros(2, 2), torch.zeros(2, 3))
        with self.assertRaises(ValueError):
            exact_forward_surrogate(torch.zeros(2, 2), torch.full((2, 2), float('nan')))


if __name__ == '__main__':
    unittest.main()
