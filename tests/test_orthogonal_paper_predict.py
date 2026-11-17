import numpy as np
import torch
from experiments.orthogonal_paper_repro.predict import predict_records


def test_predicts_with_only_features_and_frozen_weights():
    model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Identity())
    inputs = np.array([[[1, 0]], [[0, 1]], [[-1, 0]]], dtype=np.float32)
    weights = torch.tensor([[1., 0.], [0., 1.], [-1., 0.]])
    predictions = predict_records(model, weights, inputs, device='cpu', batch_size=2)
    np.testing.assert_array_equal(predictions, [0, 1, 2])
