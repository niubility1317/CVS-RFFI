import unittest

import torch

from experiments.phase1_dura_orthogonal.oracle_action_source_probe import true_action, iq_nmse


class ActionRecord:
    def __init__(self, kernel, main, image):
        self.kernel, self.main, self.image = kernel, main, image


class OracleActionTest(unittest.TestCase):
    def test_complex_channel_mapping_and_iq_normalization(self):
        kernel, coeff = torch.zeros(2, 2, 64), torch.zeros(2, 4, 256)
        kernel[:, 0, 0] = 1
        coeff[:, 0], coeff[:, 1], coeff[:, 2], coeff[:, 3] = 2, 3, 4, 5
        action = true_action(ActionRecord, kernel, coeff)
        torch.testing.assert_close(action.kernel[:, 0], torch.ones(2, dtype=torch.complex64))
        torch.testing.assert_close(action.main[0, 0], torch.tensor(2+3j))
        torch.testing.assert_close(action.image[0, 0], torch.tensor(4+5j))
        clean = torch.ones(2, 2, 256)
        torch.testing.assert_close(iq_nmse(2*clean, clean), torch.ones(2))


if __name__ == '__main__':
    unittest.main()
