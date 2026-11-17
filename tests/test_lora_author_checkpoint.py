import h5py
import numpy as np
import torch
from experiments.lora_paper_repro.checkpoint import load_author_weights, load_project_weights
from experiments.pamf_phase2_practical.vendor_extra.metric_models import LoRaExtractor


def test_loads_keras_weights_and_reorders_kernels(tmp_path):
    model = LoRaExtractor(height=26, width=6)
    layers = [model.conv1.conv, model.resblock1.conv1.conv, model.resblock1.conv2.conv,
              model.resblock2.conv1.conv, model.resblock2.conv2.conv, model.resblock3.conv1.conv,
              model.resblock3.conv2.conv, model.resblock3.shortcut.conv,
              model.resblock4.conv1.conv, model.resblock4.conv2.conv, model.fc]
    path = tmp_path / 'author.h5'
    with h5py.File(path, 'w') as f:
        for i, layer in enumerate(layers):
            name = f'conv2d_{i+1}' if i < 10 else 'dense_1'
            group = f.create_group(f'model_weights/{name}/{name}')
            shape = tuple(layer.weight.shape)
            keras_shape = (shape[2], shape[3], shape[1], shape[0]) if i < 10 else shape[::-1]
            group.create_dataset('kernel:0', data=np.full(keras_shape, i + .25, dtype=np.float32))
            group.create_dataset('bias:0', data=np.full(tuple(layer.bias.shape), i + .5, dtype=np.float32))
    load_author_weights(model, path)
    for i, layer in enumerate(layers):
        torch.testing.assert_close(layer.weight, torch.full_like(layer.weight, i + .25))
        torch.testing.assert_close(layer.bias, torch.full_like(layer.bias, i + .5))


def test_retrained_torch_checkpoint_roundtrip(tmp_path):
    original = torch.nn.Linear(4, 3)
    checkpoint = tmp_path / 'final.pt'
    torch.save({'model': original.state_dict(), 'final_epoch': 21}, checkpoint)
    restored = torch.nn.Linear(4, 3)
    load_project_weights(restored, checkpoint, device='cpu')
    for first, second in zip(original.parameters(), restored.parameters()):
        torch.testing.assert_close(first, second)
