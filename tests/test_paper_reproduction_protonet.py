import unittest

import torch


class PaperReproductionProtoNetTest(unittest.TestCase):
    def test_prototypes_are_class_means_and_euclidean_softmax_prefers_nearest(self):
        from paper_reproduction.protonet_cda.model import (
            compute_prototypes,
            distance_logits,
            prototypical_nll,
        )

        support = torch.tensor(
            [
                [0.0, 0.0],
                [2.0, 0.0],
                [8.0, 0.0],
                [10.0, 0.0],
            ]
        )
        support_labels = torch.tensor([0, 0, 1, 1])
        class_ids, prototypes = compute_prototypes(support, support_labels)

        self.assertTrue(torch.equal(class_ids, torch.tensor([0, 1])))
        self.assertTrue(torch.allclose(prototypes, torch.tensor([[1.0, 0.0], [9.0, 0.0]])))

        query = torch.tensor([[1.2, 0.0], [8.8, 0.0]])
        logits = distance_logits(query, prototypes, metric="euclidean")
        self.assertGreater(float(logits[0, 0]), float(logits[0, 1]))
        self.assertGreater(float(logits[1, 1]), float(logits[1, 0]))

        loss, pred = prototypical_nll(support, support_labels, query, torch.tensor([0, 1]))
        self.assertLess(float(loss), 0.01)
        self.assertTrue(torch.equal(pred, torch.tensor([0, 1])))

    def test_episode_validator_rejects_leakage_and_wrong_k_shot(self):
        from paper_reproduction.common.episodes import EpisodeBatch, validate_closed_set_episode

        valid = EpisodeBatch(
            support_x=torch.randn(4, 2),
            support_y=torch.tensor([3, 3, 7, 7]),
            query_x=torch.randn(2, 2),
            query_y=torch.tensor([3, 7]),
            support_ids=("s0", "s1", "s2", "s3"),
            query_ids=("q0", "q1"),
        )
        validate_closed_set_episode(valid, k_shot=2)

        leaked = EpisodeBatch(
            support_x=valid.support_x,
            support_y=valid.support_y,
            query_x=valid.query_x,
            query_y=valid.query_y,
            support_ids=("same", "s1", "s2", "s3"),
            query_ids=("same", "q1"),
        )
        with self.assertRaisesRegex(ValueError, "support/query leakage"):
            validate_closed_set_episode(leaked, k_shot=2)

        wrong_k = EpisodeBatch(
            support_x=valid.support_x[:3],
            support_y=torch.tensor([3, 3, 7]),
            query_x=valid.query_x,
            query_y=valid.query_y,
            support_ids=("s0", "s1", "s2"),
            query_ids=valid.query_ids,
        )
        with self.assertRaisesRegex(ValueError, "K-shot"):
            validate_closed_set_episode(wrong_k, k_shot=2)


if __name__ == "__main__":
    unittest.main()
