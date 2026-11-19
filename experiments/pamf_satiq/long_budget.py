"""Run the full-header PAMF arm until its Oxford validation target is reached."""

import argparse
from pathlib import Path
import subprocess
import sys

from .header_experiment import reaches_threshold, train
from .prepare import write_json
from .run import read


ARM = "pamf_full"


def launch(config, compact, full, out):
    rule = read(config)
    percentage = rule["target_accuracy_percent"]
    if (rule["schema"] != "pamf_satiq_long_budget_v1"
            or type(percentage) is not int or not 1 <= percentage <= 100
            or not isinstance(rule["checkpoint_period"], int) or rule["checkpoint_period"] < 1):
        raise ValueError("Invalid source stopping rule")
    base = config.parent / rule["base_config"]
    cfg = read(base)
    cfg.update(epochs=None, target_accuracy_percent=percentage,
               checkpoint_epochs=rule["checkpoint_epochs"], checkpoint_period=rule["checkpoint_period"])
    out.mkdir(exist_ok=False)
    arm_out = out / ARM
    train(ARM, compact, full, cfg, arm_out)
    verdict = read(arm_out / "threshold_reached.json")
    if (verdict["threshold_percent"] != percentage
            or not reaches_threshold(verdict["correct"], verdict["total"], percentage)):
        raise ValueError("Swiss inference requires the frozen UK threshold checkpoint")
    write_json(out / "uk_threshold.json", verdict)
    subprocess.run([sys.executable, "-B", "-u", "-m", "experiments.pamf_satiq.header_experiment",
                    "predict", str(base), str(compact), str(full), str(arm_out), "--arm", ARM], check=True)
    subprocess.run([sys.executable, "-B", "-u", "-m", "experiments.pamf_satiq.score",
                    str(compact / "source/contract.json"), str(full / "target/query/ids.npy"),
                    str(compact / "target/scorer/truth.npz"), str(out), "--arms", ARM], check=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("compact", type=Path)
    parser.add_argument("full", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    launch(args.config, args.compact, args.full, args.out)
