"""Compare local author Keras models with weight-mapped PyTorch ports."""
import argparse
import json
import os
from pathlib import Path
import sys

os.environ['KERAS_BACKEND'] = 'torch'
os.environ['CUDA_VISIBLE_DEVICES'] = ''  # Both reference and port use CPU float32.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--compat-root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    import numpy as np
    import torch
    import keras
    from keras.src.backend.torch.core import device_scope
    cpu_scope = device_scope('cpu')
    cpu_scope.__enter__()
    from baselines.common.li_backbones import PosterHomegrown, RadioNetDF
    torch.set_num_threads(2)
    torch.backends.mkldnn.enabled = False
    torch.set_float32_matmul_precision('highest')
    sys.path[:0] = [str(a.compat_root / 'RadioFingerprinting/fine-tuning'),
                   str(a.compat_root / 'RadioNet/models')]
    import radioConv
    import rf_models
    result = {}
    for paper in ['poster', 'radionet']:
        keras.utils.set_random_seed(9123)
        keras.config.set_image_data_format('channels_first' if paper == 'poster' else 'channels_last')
        reference = (radioConv.create_model('homegrown', (2, 256), 6, channel='first') if paper == 'poster'
                     else rf_models.create_DF((256, 2), class_num=6, classification=True))
        port = (PosterHomegrown(6) if paper == 'poster' else RadioNetDF(6)).eval()
        if paper == 'poster':
            pairs = [(port.conv1, 'block1_conv1'), (port.conv2, 'block2_conv1'),
                     (port.dense1, 'dense1'), (port.dense2, 'dense2'), (port.head, 'dense3')]
        else:
            pairs = [(port.blocks[b][j], f'block{b + 1}_conv{k}')
                     for b in range(4) for j, k in [(0, 1), (2, 2)]] + [(port.head, 'FeaturesVec')]
        with torch.no_grad():
            for layer, name in pairs:
                w, bias = reference.get_layer(name).get_weights()
                w = w.transpose(2, 1, 0) if w.ndim == 3 else w.T
                layer.weight.copy_(torch.tensor(w.tolist()))
                layer.bias.copy_(torch.tensor(bias.tolist()))
        rng = np.random.default_rng(7131)
        x = rng.standard_normal((7, 2, 256)).astype('float32')
        expected = reference(x if paper == 'poster' else x.transpose(0, 2, 1), training=False)
        expected = expected.detach().cpu()
        actual = port(torch.tensor(x.tolist())).softmax(-1).detach().cpu()
        max_error = float((expected - actual).abs().max())
        if not torch.allclose(expected, actual, atol=2e-6, rtol=2e-5):
            if paper == 'radionet':
                pp = torch.tensor(x.tolist())
                rr = torch.tensor(x.transpose(0, 2, 1).copy().tolist())
                for b in range(4):
                    names = [f'block{b+1}_conv1', f'block{b+1}_adv_act1' if b==0 else f'block{b+1}_act1',
                             f'block{b+1}_conv2', f'block{b+1}_adv_act2' if b==0 else f'block{b+1}_act2',
                             f'block{b+1}_pool'] + ([f'block{b+1}_dropout'] if b<3 else [])
                    for layer, name in zip(port.blocks[b], names):
                        pp = layer(pp)
                        rr = reference.get_layer(name)(rr, training=False)
                        print(name, float((pp-rr.transpose(1,2)).abs().max().detach()))
                cur = torch.tensor(x.tolist())
                for b in range(4):
                    cur = port.blocks[b](cur)
                    layer_name = f'block{b + 1}_dropout' if b < 3 else 'block4_pool'
                    probe = keras.Model(reference.input, reference.get_layer(layer_name).output)
                    other = probe(x.transpose(0, 2, 1), training=False).detach().cpu().transpose(1, 2)
                    print(layer_name, float((cur - other).abs().max()), cur.shape, other.dtype)
            raise AssertionError(f'{paper}: probability mismatch {max_error}')
        before = {k: v.clone() for k, v in port.state_dict().items()}
        port.author_finetune(9)
        frozen = {k: v.clone() for k, v in port.state_dict().items() if k in before
                  and (k.startswith('conv') if paper == 'poster' else k.startswith('blocks'))}
        opt = torch.optim.Adam([p for p in port.parameters() if p.requires_grad], lr=.001)
        opt.zero_grad()
        torch.nn.functional.cross_entropy(port(torch.tensor(x.tolist())), torch.arange(7) % 9).backward()
        opt.step()
        assert all(torch.equal(v, port.state_dict()[k]) and torch.equal(v, before[k]) for k, v in frozen.items())
        result[paper] = dict(max_probability_abs_error=max_error, frozen_prefix_unchanged=True,
                             trainable=[k for k, v in port.named_parameters() if v.requires_grad])
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
