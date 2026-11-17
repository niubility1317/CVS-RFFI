"""Load the published Keras Extractor_1 weights into the audited Torch port."""
import h5py
import numpy as np
import torch


def load_author_weights(model, path):
    layers = [model.conv1.conv, model.resblock1.conv1.conv, model.resblock1.conv2.conv,
              model.resblock2.conv1.conv, model.resblock2.conv2.conv,
              model.resblock3.conv1.conv, model.resblock3.conv2.conv, model.resblock3.shortcut.conv,
              model.resblock4.conv1.conv, model.resblock4.conv2.conv, model.fc]
    with h5py.File(path, 'r') as source:
        for index, layer in enumerate(layers):
            name = f'conv2d_{index + 1}' if index < 10 else 'dense_1'
            group = source['model_weights'][name][name]
            weights, bias = group['kernel:0'][:], group['bias:0'][:]
            weights = np.transpose(weights, (3, 2, 0, 1)) if index < 10 else weights.T
            if tuple(weights.shape) != tuple(layer.weight.shape) or tuple(bias.shape) != tuple(layer.bias.shape):
                raise ValueError(f'author layer shape mismatch: {name}')
            if not np.isfinite(weights).all() or not np.isfinite(bias).all():
                raise ValueError(f'nonfinite author weights: {name}')
            with torch.no_grad():
                layer.weight.copy_(torch.from_numpy(weights.copy()).to(layer.weight.device))
                layer.bias.copy_(torch.from_numpy(bias.copy()).to(layer.bias.device))


def load_project_weights(model, path, *, device):
    state = torch.load(path, map_location=device, weights_only=False)
    if not isinstance(state, dict) or 'model' not in state or 'final_epoch' not in state:
        raise ValueError('not a completed project training checkpoint')
    model.load_state_dict(state['model'], strict=True)
    if not all(bool(torch.isfinite(param).all()) for param in model.parameters()):
        raise ValueError('nonfinite project checkpoint weights')
