"""Reproduce a failed D92 split using its labeled support and ground only."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "tools"), str(ROOT / "code")]
import numpy as np
import torch
from cvs_d92_matched import read, registered_features, fit_d92, load_ground


def diagnostic(args):
    row, capsule, out = args.row, args.capsule, args.output
    if out.exists():
        raise FileExistsError(out)
    last = None
    for line in (row / "D92_PREDICT.log").read_text(encoding="utf-8").splitlines():
        try:
            record = json.loads(line)
        except ValueError:
            continue
        if "split_id" in record:
            last = record["split_id"]
    paths = sorted((capsule / "splits").glob("*.json"))
    if args.split:
        path = capsule / "splits" / (args.split + ".json")
    else:
        position = next(i for i, path in enumerate(paths) if path.stem == last) + 1
        path = paths[position]
    split = read(path)
    support = split["support_indices"]
    with np.load(row / "received_features/received_features.npz", allow_pickle=False) as data:
        identity = data["identity160"][support]
        feature_ids = data["ids"][support]
    with np.load(capsule / "received.npz", allow_pickle=False) as data:
        iq = data["iq"][support]
        ids = data["ids"][support]
    if not np.array_equal(ids, feature_ids):
        raise ValueError("Support identity mismatch")
    x = registered_features(iq, identity)
    ground = load_ground(row / "ground")
    out.mkdir(parents=True, exist_ok=False)
    report = dict(split_id=split["split_id"], seed=args.seed, k=split["k"],
        classes=len(split["registered_classes"]), support_count=len(support),
        query_features_read=False, query_truth_read=False, ground_schema="v2_rank3_radius",
        numpy=np.__version__, torch=torch.__version__)
    try:
        state = fit_d92(support_features=x, support_labels=split["support_labels"],
                       classes=split["registered_classes"], ground=ground, seed=args.seed)
        report.update(status="FIT_PASSED", active_feature_dim=state.active_feature_dim)
    except Exception as exc:
        report.update(status="FIT_FAILED", error_type=type(exc).__name__, error=str(exc))
        traceback = exc.__traceback__
        captured = None
        while traceback is not None:
            if traceback.tb_frame.f_code.co_name == "_fit_equal_prior_lda":
                captured = traceback.tb_frame.f_locals
            traceback = traceback.tb_next
        if captured is None:
            raise
        data = {name: captured[name] for name in (
            "transformed", "targets", "coefficients64", "intercept64", "coefficients", "intercept",
            "covariance", "fitted_means") if name in captured}
        data.update(class_count=np.asarray(captured["class_count"]), k_shot=np.asarray(captured["k_shot"]))
        estimator = captured.get("estimator")
        if estimator is not None:
            rows = np.asarray(data["transformed"], dtype=np.float64)
            sklearn_scores = estimator.decision_function(rows)
            score64 = rows @ data["coefficients64"].T + data["intercept64"]
            score32 = rows @ data["coefficients"].astype(np.float64).T + data["intercept"].astype(np.float64)
            target = np.asarray(estimator.predict(rows))
            data.update(sklearn_coefficients=estimator.coef_, sklearn_intercept=estimator.intercept_)
            mismatches = np.flatnonzero(score32.argmax(1) != target)
            report.update(inner_fit_class_count=int(captured["class_count"]), inner_fit_k=int(captured["k_shot"]),
                covariance_condition=float(np.linalg.cond(data["covariance"])),
                coefficient_max_abs=float(np.abs(data["coefficients64"]).max()),
                mismatch_support_indices=mismatches.tolist(),
                float64_vs_sklearn_mismatches=int(np.sum(score64.argmax(1) != target)),
                float32_vs_sklearn_mismatches=int(len(mismatches)),
                coefficient64_vs_sklearn_max_abs=float(np.abs(data["coefficients64"] - estimator.coef_).max()),
                score32_vs_64_max_abs=float(np.abs(score32 - score64).max()),
                sklearn_margins_on_mismatches=[float(np.sort(sklearn_scores[i])[-1] - np.sort(sklearn_scores[i])[-2]) for i in mismatches])
        np.savez_compressed(out / "failed_inner_support_fit.npz", **data)
    (out / "diagnostic.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("row", "capsule", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--split")
    torch.set_num_threads(1)
    diagnostic(parser.parse_args())


if __name__ == "__main__":
    main()
