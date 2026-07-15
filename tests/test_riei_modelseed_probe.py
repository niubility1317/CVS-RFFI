from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "baselines" / "common" / "cvs_data.py"
QUEUE = ROOT / "run_wisig_paper_scope_queue.sh"
LAUNCHER = ROOT / "code" / "scripts" / "launch_riei_modelseed_probe_20260715.sh"


def test_split_seed_is_backward_compatible_and_traceable() -> None:
    text = DATA.read_text(encoding="utf-8")
    assert '"--wisig_split_seed"' in text
    assert 'if split_seed < 0:' in text
    assert 'split_seed = int(getattr(args, "seed", 1337))' in text
    assert text.count("seed=split_seed,") == 5
    assert '"model_seed": int(getattr(args, "seed", 1337))' in text
    assert '"split_seed": split_seed' in text


def test_paper_queue_forwards_independent_split_seed() -> None:
    text = QUEUE.read_text(encoding="utf-8")
    assert 'WISIG_SPLIT_SEED="${WISIG_SPLIT_SEED:-${SEED}}"' in text
    assert '--wisig_split_seed "${WISIG_SPLIT_SEED}"' in text


def test_modelseed_probe_is_fixed_partition_and_capacity_gated() -> None:
    text = LAUNCHER.read_text(encoding="utf-8")
    assert "MODEL_SEEDS=(0 42)" in text
    assert '"WISIG_SPLIT_SEED=${SPLIT_SEED}"' in text
    assert '"SEED=${model_seed}"' in text
    assert text.count('control_seed1337_last10') >= 2
    assert '[[ "${#GPU_IDS[@]}" -eq 8 ]]' in text
    assert "planned_peak=1" in text
    assert "total <= MAX_TRAIN_PER_GPU" in text
