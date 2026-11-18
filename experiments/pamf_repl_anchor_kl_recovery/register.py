"""Register an identical matrix and the lower validation memory concurrency."""
import argparse
import json
from pathlib import Path

from experiments.pamf_repl_anchor_kl import register as anchor
from . import plan


def make(template):
    old = anchor.plan
    anchor.plan = plan
    try:
        result = anchor.make(template)
    finally:
        anchor.plan = old
    result.update(display_name='定向 KL 矩阵技术恢复',
                  description='复用上一批已完成的同一 E50，只补训 OOM 和未启动行，再完成独立评分。',
                  replaces_run_id=plan.PREVIOUS_RUN)
    result['execution'].update(
        gpu_policy='GPU 0–6 only; maximum two CUDA processes per card including other owners; minimum admission 12 GiB',
        local_artifact_root='local_artifacts/pamf_repl_anchor_kl_recovery_20261118',
        launch_command='python -m experiments.pamf_repl_anchor_kl_recovery.publish --repo REPO --output OUTPUT --context CONTEXT --ssh-config CONFIG')
    result['notes'].extend([
        'Previous controller must reach PARTIAL_TECHNICAL_FAILURE after all healthy workers exit.',
        'Frozen rows reused only by exact source contract, model seed, recipe, parent and blind observations. No performance selection.',
        'Terminal source OOM rows and queued rows get independent E20-to-E50 execution at the same seed, doses and budget.'])
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--template', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    result = make(json.loads(Path(args.template).read_text(encoding='utf-8-sig')))
    Path(args.output).write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
