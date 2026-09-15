"""Verify the author's frozen prefix and saved prediction path after integration tests."""
import argparse
import json
import os
from pathlib import Path
os.environ['KERAS_BACKEND'] = 'torch'
import keras
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    args = parser.parse_args()
    results = {}
    for paper in ['poster', 'radionet']:
        keras.config.set_image_data_format('channels_first' if paper == 'poster' else 'channels_last')
        source = keras.models.load_model(args.root / f'{paper}_train/final.keras', compile=False)
        tuned = keras.models.load_model(args.root / f'{paper}_tune/final.keras', compile=False)
        for a, b in zip(source.layers[:-3], tuned.layers[:-3]):
            if b.weights:
                assert not b.trainable, b.name
            for wa, wb in zip(a.get_weights(), b.get_weights()):
                np.testing.assert_array_equal(wa, wb)
        with np.load(args.root / 'query.npz') as data:
            x = data['x']
        if paper == 'poster':
            x = x.transpose(0, 2, 1)
        single = np.concatenate([tuned.predict(v[None], verbose=0) for v in x])
        with np.load(args.root / f'{paper}_predict/predictions.npz') as data:
            np.testing.assert_allclose(single, data['probabilities'], atol=2e-5, rtol=2e-5)
        results[paper] = dict(frozen_prefix_exact=True, per_sample_reload_prediction_match=True,
                              trainable_layers=[v.name for v in tuned.layers if v.trainable and v.weights])
    (args.root / 'weight_checks.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
