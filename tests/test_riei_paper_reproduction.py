from pathlib import Path
import re
import subprocess


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "baselines" / "common" / "cvs_data.py"
QUEUE = ROOT / "run_wisig_paper_scope_queue.sh"
LAUNCHER = ROOT / "code" / "scripts" / "launch_riei_paper_reproduction.sh"


def test_split_seed_is_backward_compatible_and_traceable() -> None:
    text = DATA.read_text(encoding="utf-8")
    assert '"--wisig_split_seed"' in text
    assert "if split_seed < 0:" in text
    assert 'split_seed = int(getattr(args, "seed", 1337))' in text
    assert text.count("seed=split_seed,") == 5
    assert '"model_seed": int(getattr(args, "seed", 1337))' in text
    assert '"split_seed": split_seed' in text


def test_paper_queue_forwards_independent_split_seed() -> None:
    text = QUEUE.read_text(encoding="utf-8")
    assert 'WISIG_SPLIT_SEED="${WISIG_SPLIT_SEED:-${SEED}}"' in text
    assert '--wisig_split_seed "${WISIG_SPLIT_SEED}"' in text


def test_unique_launcher_fixes_final_scientific_configuration() -> None:
    text = LAUNCHER.read_text(encoding="utf-8")
    assert "readonly MODEL_SEED=42" in text
    assert "readonly SPLIT_SEED=1337" in text
    assert "readonly FED_VARIANT=short_stem1d" in text
    assert "--seed)" not in text
    assert "--split-seed)" not in text
    for setting in (
        '"BASELINE_EPOCHS=200"',
        '"RIEI_PAPER_EVAL_LAST_N=10"',
        '"RIEI_OPTIMIZER=sgd"',
        '"RIEI_SGD_MOMENTUM=0"',
        '"RIEI_CE_REDUCTION=mean"',
        '"RIEI_MI_REDUCTION=mean"',
        '"RIEI_IE_REDUCTION=mean"',
        '"RIEI_WISIG_RMS_NORMALIZE=0"',
        '"RIEI_LAMBDA_FEATURE_NORM=0"',
        '"RIEI_FED_VARIANT=${FED_VARIANT}"',
    ):
        assert setting in text


def test_unique_launcher_covers_all_table3_rows_and_capacity_gate() -> None:
    text = LAUNCHER.read_text(encoding="utf-8")
    rows = re.findall(
        r'^\s+"(\d+)\|([^|]+)\|([^|]+)\|([^|]+)\|([0-9.]+)\|([0-9.]+)"$',
        text,
        flags=re.MULTILINE,
    )
    assert len(rows) == 12
    assert [int(row[0]) for row in rows] == list(range(1, 13))
    assert '[[ "${#GPU_IDS[@]}" -eq 8 ]]' in text
    assert "planned_peak=1" in text
    assert "total <= MAX_TRAIN_PER_GPU" in text
    assert "unique run/log root already exists" in text


def test_unique_launcher_dry_run_emits_twelve_fixed_jobs() -> None:
    proc = subprocess.run(
        ["bash", "code/scripts/launch_riei_paper_reproduction.sh", "--dry-run", "--run-id", "riei_final_dry_run"],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    out = proc.stdout + proc.stderr
    assert proc.returncode == 0, out
    assert out.count("[JOB]") == 12
    assert out.count("[CAPACITY]") == 8
    assert out.count("SEED=42") == 12
    assert out.count("WISIG_SPLIT_SEED=1337") == 12
    assert out.count("RIEI_FED_VARIANT=short_stem1d") == 12
