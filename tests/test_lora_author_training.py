import numpy as np
import torch
from experiments.lora_paper_repro.train import sample_triplets, triplet_loss
from experiments.pamf_phase2_practical.vendor_extra.metric_models import LoRaExtractor


def test_triplets_sample_within_and_across_source_classes():
    labels = np.repeat(np.arange(3), 5)
    groups = [np.flatnonzero(labels == i) for i in range(3)]
    anchor, positive, negative = sample_triplets(groups, 300, np.random.default_rng(82))
    assert np.all(labels[anchor] == labels[positive])
    assert np.all(labels[anchor] != labels[negative])


def test_triplet_loss_uses_author_squared_distance_margin():
    anchor = torch.tensor([[1., 0.], [1., 0.]])
    positive = torch.tensor([[1., 0.], [0., 1.]])
    negative = torch.tensor([[0., 1.], [1., 0.]])
    torch.testing.assert_close(triplet_loss(anchor, positive, negative), torch.tensor([0., 2.1]))


def test_concatenated_triplet_forward_matches_separate_forwards():
    torch.manual_seed(3)
    model = LoRaExtractor(height=26, width=6)
    batches = [torch.randn(2, 1, 26, 6) for _ in range(3)]
    combined = model(torch.cat(batches)).split(2)
    separate = [model(batch) for batch in batches]
    for first, second in zip(combined, separate):
        torch.testing.assert_close(first, second, atol=1e-6, rtol=1e-5)
