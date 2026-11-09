"""Freeze both raw target populations before the independent CPU scorer."""
from pathlib import Path
import time

import numpy as np

from experiments.cric_dura_comparison.io import read, write, tensor
from experiments.cric_dura_comparison.evaluation import handle, reject_truth
from experiments.practical_full_2x2.plan import VIEWS
from .method import TASKS, READOUTS, load_reader


def predict(c, device="cuda:0"):
    import torch
    reject_truth(c)
    if list(c["tests"]) != list(TASKS) or c["readouts"] != READOUTS:
        raise ValueError("Only both declared populations and raw readout are permitted")
    contract = read(c["source_contract"])
    reader, _ = load_reader(c["source_output"], contract, device)
    out = Path(c["output_root"])
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    counts = {}
    for task in TASKS:
        manifest, ids, positions = handle(c["tests"][task]["views_root"], c["classes"])
        expected = {"cross_rx": 126000, "cross_both": 42000}[task]
        if len(ids) != expected:
            raise ValueError("Declared target population count differs")
        values = {}
        for view in VIEWS:
            path = manifest["clean_ref"] if view == "clean" else str(
                Path(manifest.get("observations_root", c["tests"][task]["views_root"])) / (view + ".npy"))
            array = np.load(path, mmap_mode="r", allow_pickle=False)
            if array.ndim != 3 or array.shape[1:] != (2, 256):
                raise ValueError("Received IQ shape differs")
            result = []
            with torch.inference_mode():
                for begin in range(0, len(ids), 128):
                    ix = positions[begin:begin + 128] if positions is not None else slice(begin, begin + 128)
                    x = tensor(np.ascontiguousarray(array[ix], np.float32), device)
                    logits = reader.classify_features(reader.features(x))
                    if logits.shape != (len(x), 6) or not bool(torch.isfinite(logits).all()):
                        raise FloatingPointError("Invalid default-head logits")
                    result.extend(logits.argmax(1).cpu().tolist())
            values["raw__" + view] = np.asarray(result, np.int64)
            print("FROZEN_PREDICTION", c["row_id"], task, view, len(ids), flush=True)
        folder = out / task
        folder.mkdir()
        np.savez(folder / "predictions.npz", ids=ids, **values)
        counts[task] = dict(count=len(ids), views=list(VIEWS), readouts=READOUTS)
    write(out / "complete.json", dict(status="ALL_PREDICTIONS_COMPLETE", run_id=c["run_id"],
          row_id=c["row_id"], tests=counts, views=list(VIEWS), readouts=READOUTS,
          truth_read=False, query_fit=False, seconds=time.monotonic() - started))


def score(c):
    from experiments.cric_dura_comparison import evaluation as old
    old.tasks = lambda _: list(TASKS)
    old.score(c)
