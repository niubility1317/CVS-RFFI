from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]


def read(relpath: str) -> str:
    return (ROOT / relpath).read_text(encoding="utf-8")


def test_root_queue_protocols_and_claim_boundaries():
    text = read("scripts/launchers/run_cvs_baseline_queue.sh")

    assert "cvs_day_rx|riei_original|drift_day1" in text
    assert 'SAT_VIEW_AUG="${SAT_VIEW_AUG:-0}"' in text
    assert "--sat-view-aug" in text
    assert 'RIEI_PAPER_EVAL_LAST_N="${RIEI_PAPER_EVAL_LAST_N:-10}"' in text
    assert 'DRIFT_PAPER_EVAL_LAST_N="${DRIFT_PAPER_EVAL_LAST_N:-5}"' in text
    assert "baseline_paper_audit_${WISIG_PROTOCOL}_seed${SEED}" in text


def test_riei_table3_wrapper_expands_all_paper_rows():
    text = read("baselines/scripts/run_riei_original_table3_queue.sh")
    rows = re.findall(r'"rx[^"]+\|[^"]+\|[^"]+"', text)

    assert len(rows) == 12
    assert "--wisig-protocol" in text
    assert "riei_original" in text
    assert "RIEI_PAPER_EVAL_LAST_N=10" in text
    assert 'DRY_RUN="${DRY_RUN:-1}"' in text


def test_docs_record_status_and_dry_run_not_success():
    audit = read("baselines/PAPER_CODE_AUDIT.md")
    baselines_readme = read("baselines/README.md")
    root_readme = read("README.md")

    assert "2026-06-26落实状态" in audit
    assert "不采纳" in audit
    assert "待真实训练验证" in audit
    assert "scripts/launchers/run_cvs_baseline_queue.sh" in baselines_readme
    assert "baselines/scripts/run_riei_original_table3_queue.sh" in baselines_readme
    assert "do not by themselves constitute" in baselines_readme
    assert "RIEI original" in root_readme
