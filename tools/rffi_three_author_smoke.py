"""Run downloaded author model definitions on Keras 3's PyTorch backend.

This is compatibility validation, NOT an original-data numerical reproduction.
No upstream file is modified. Only selected model-building functions are loaded.
"""
from __future__ import annotations

import argparse
import ast
import json
import os
from pathlib import Path

os.environ.setdefault("KERAS_BACKEND", "torch")

import keras
import numpy as np


def author_model(root: Path, paper: str, classes: int = 5):
    if paper == "poster":
        path = root / "RadioFingerprinting/fine-tuning/radioConv.py"
        names = {"createHomegrown"}
    else:
        path = root / "RadioNet/models/rf_models.py"
        names = {"create_DF"}
    tree = ast.parse(path.read_text(encoding="utf-8-sig"))
    selected = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    if len(selected) != 1:
        raise ValueError(f"Author function missing: {path}")
    namespace = {name: getattr(keras.layers, name) for name in dir(keras.layers) if not name.startswith("_")}
    namespace["Model"] = keras.Model
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec"), namespace)
    if paper == "poster":
        # Explicit channels-last fixes the original Conv/GAP format inconsistency.
        return namespace["createHomegrown"]((288, 2), classes, "channels_last", False)
    return namespace["create_DF"]((288, 2), classes, 64, True)


def prepare_adaptation(source, root: Path, paper: str, policy: str):
    if policy == "paper_last_layer":
        model = keras.models.clone_model(source)
        model.set_weights(source.get_weights())
        for layer in model.layers[:-1]:
            layer.trainable = False
    elif policy == "author_last_three":
        # Exact copying/freezing range used by both author finetune.py files.
        model = author_model(root, paper, source.output_shape[-1])
        for target_layer, source_layer in zip(model.layers[:-3], source.layers[:-3]):
            target_layer.set_weights(source_layer.get_weights())
            target_layer.trainable = False
    else:
        raise ValueError(policy)
    # Compile AFTER trainability changes (required by current Keras).
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="sparse_categorical_crossentropy")
    return model


def validate(root: Path, out: Path):
    out.mkdir(parents=True, exist_ok=False)
    keras.utils.set_random_seed(9152026)
    rng = np.random.default_rng(9152026)
    x = rng.normal(size=(10, 288, 2)).astype("float32")
    y = np.arange(10) % 5
    records = []
    for paper in ("poster", "radionet"):
        source = author_model(root, paper)
        source.compile(optimizer="adam", loss="sparse_categorical_crossentropy")
        source_loss = float(source.train_on_batch(x, y))
        for policy in ("paper_last_layer", "author_last_three"):
            model = prepare_adaptation(source, root, paper, policy)
            frozen = [(layer, [w.copy() for w in layer.get_weights()]) for layer in model.layers if not layer.trainable]
            trainable_before = [keras.ops.convert_to_numpy(w).copy() for w in model.trainable_weights]
            loss = float(model.train_on_batch(x, y))
            unchanged = all(np.array_equal(a, b) for layer, old in frozen for a, b in zip(old, layer.get_weights()))
            changed = any(not np.array_equal(a, keras.ops.convert_to_numpy(b)) for a, b in zip(trainable_before, model.trainable_weights))
            predictions = model.predict(x, verbose=0)
            single = np.concatenate([model.predict(row[None], verbose=0) for row in x])
            destination = out / f"{paper}_{policy}.keras"
            model.save(destination)
            loaded = keras.models.load_model(destination)
            restored = loaded.predict(x, verbose=0)
            assert np.isfinite([source_loss, loss]).all() and np.isfinite(predictions).all()
            assert unchanged and changed, (paper, policy, unchanged, changed)
            np.testing.assert_allclose(predictions, single, atol=2e-5, rtol=2e-5)
            np.testing.assert_allclose(predictions, restored, atol=1e-6, rtol=1e-6)
            records.append(dict(paper=paper, policy=policy, parameters=model.count_params(),
                                source_loss=source_loss, adaptation_loss=loss,
                                frozen_weights_unchanged=unchanged, trainable_weights_changed=changed,
                                per_sample_inference=True, checkpoint_roundtrip=True))
            print(paper, policy, "PASS", flush=True)
    result = dict(status="COMPATIBILITY_SMOKE_ONLY", data="synthetic", keras=keras.__version__,
                  backend=keras.backend.backend(), records=records)
    (out / "validation.json").write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    validate(args.sources, args.out)
