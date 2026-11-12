"""Release the staged reduced-weight Hybrid queue on N607."""
import argparse

from experiments.pamf_branch_training import publish as base
from .hybrid_weight_plan import RUN
from .source_tuning_publish import publish


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for option in ('repo', 'output', 'context', 'ssh-config'):
        parser.add_argument('--' + option, required=True)
    args = parser.parse_args()
    old = base.REMOTE
    base.REMOTE = old.replace("'dispatch','--config'", "'stage','--config'")
    base.REMOTE = base.REMOTE.replace("result=dict(status='SUBMITTED_NOT_YET_VERIFIED'",
        "if child.wait()!=0:raise RuntimeError('Queue preparation failed; inspect controller.log')\nresult=dict(status='STAGED_NOT_YET_VERIFIED'")
    try:
        publish(args.repo, args.output, args.context, args.ssh_config, run_id=RUN,
                protocol_id='ManyRx', setting='ten_class',
                worker_module='experiments.pamf_physics_tri.hybrid_weight_runner',
                gpus=(0, 1, 3, 4, 5, 7, 2, 6), max_total_processes_per_gpu=6)
    finally:
        base.REMOTE = old
