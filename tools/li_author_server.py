"""Server interface to the author's supervised CNN methods; no network rewrite.

NPZ interface: x=(N,L,2) received IQ, y=int labels, classes=ordered string IDs.
Prediction inputs have no labels. Splitting/preprocessing remain caller duties.
"""
import argparse
import importlib
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--compat-root', type=Path, required=True)
    p.add_argument('--paper', choices=['poster', 'radionet'], required=True)
    p.add_argument('--stage', choices=['train', 'tune', 'predict'], required=True)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path)
    p.add_argument('--backend', choices=['torch', 'tensorflow'], default='torch')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--epochs', type=int)
    p.add_argument('--batch-size', type=int)
    args = p.parse_args()
    if args.stage != 'train' and args.checkpoint is None:
        p.error('--checkpoint is required for tune/predict')
    if args.stage == 'train' and args.checkpoint is not None:
        p.error('source training starts from scratch')
    if args.output.exists():
        p.error('output already exists; use a fresh directory')
    os.environ['KERAS_BACKEND'] = args.backend
    os.environ.setdefault('MPLBACKEND', 'Agg')
    import numpy as np
    import keras
    keras.utils.set_random_seed(args.seed)
    poster = args.paper == 'poster'
    keras.config.set_image_data_format('channels_first' if poster else 'channels_last')
    folder = args.compat_root / ('RadioFingerprinting/fine-tuning' if poster else 'RadioNet')
    sys.path[:0] = [str(folder.resolve()), str((folder / 'models').resolve())]
    with np.load(args.data, allow_pickle=False) as data:
        x = np.asarray(data['x'], dtype='float32')
        classes = data['classes'].astype(str).tolist()
        if x.ndim != 3 or x.shape[-1] != 2 or not np.isfinite(x).all():
            raise ValueError('x must be finite (N,L,2) IQ; no automatic slicing/normalization')
        if len(classes) < 5 or len(set(classes)) != len(classes):
            raise ValueError('unique ordered class IDs required; original top-5 needs >=5 classes')
        y = None
        if args.stage != 'predict':
            labels = data['y']
            if labels.shape != (len(x),) or labels.dtype.kind not in 'iu' or labels.min() < 0 or labels.max() >= len(classes):
                raise ValueError('invalid integer labels')
            y = keras.utils.to_categorical(labels, len(classes))
        elif 'y' in data.files:
            raise ValueError('predict input must not contain truth labels')
    if poster:
        x = x.transpose(0, 2, 1)
    if args.checkpoint:
        meta = json.loads(args.checkpoint.with_suffix('.json').read_text(encoding='utf-8'))
        if meta['classes'] != classes or meta['paper'] != args.paper or meta['input_shape'] != list(x.shape[1:]):
            raise ValueError('checkpoint class/model/input contract mismatch')
    args.output.mkdir(parents=True)
    if args.stage == 'predict':
        model = keras.models.load_model(args.checkpoint, compile=False)
        prediction = model.predict(x, batch_size=args.batch_size or 100, verbose=0)
        np.savez(args.output / 'predictions.npz', probabilities=prediction, classes=np.array(classes))
    else:
        module = importlib.import_module('finetune')
        module.modelDir = str(args.output.resolve())
        module.ResDir = str(args.output.resolve())
        opts = SimpleNamespace(verbose=1, trainData=str(args.data), tuneData=str(args.data),
                               modelType='homegrown' if poster else 'DF', D2=False)
        module.opts = opts
        cnn = module.CNN(opts) if poster else module.CNN(opts, SimpleNamespace(sample=False))
        cnn.input_shape = tuple(x.shape[1:])
        cnn.emb_size = len(classes)
        if args.epochs is not None:
            if args.epochs < 1:
                raise ValueError('epochs must be positive')
            cnn.trainEpochs = cnn.tuneEpochs = args.epochs
        if args.batch_size is not None:
            cnn.batch_size = args.batch_size
        cnn.trainModelPath = str(args.checkpoint or (args.output / 'source.keras'))
        cnn.tuneModelPath = str(args.output / 'target.keras')
        if args.stage == 'train':
            # Author training method; its reporting evaluation uses source data only.
            cnn.train(x, y, x, y, len(classes))
            model = keras.models.load_model(cnn.trainModelPath, compile=False)
        else:
            # RadioNet overwrites tuneModelPath internally; make checkpoint suffix API-compatible.
            original = module.ModelCheckpoint
            def checkpoint(*a, **kw):
                kw['filepath'] = str(args.output / 'target.keras')
                return original(*a, **kw)
            module.ModelCheckpoint = checkpoint
            model = cnn.tuneTheModel(x, y, len(classes))
        model.save(args.output / 'final.keras')
        metadata = dict(paper=args.paper, classes=classes, input_shape=list(x.shape[1:]),
                        stage=args.stage, backend=args.backend, seed=args.seed,
                        parent=str(args.checkpoint) if args.checkpoint else None,
                        data=str(args.data.resolve()), core='author_unchanged',
                        selection='source_best_val' if args.stage == 'train' else 'author_final_epoch')
        (args.output / 'final.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    (args.output / 'completed.json').write_text(json.dumps(dict(stage=args.stage, samples=len(x))), encoding='utf-8')


if __name__ == '__main__':
    main()
