"""Reuse the established one-archive transfer with all eight user-authorized GPUs."""
import argparse
import json
from pathlib import Path
from experiments.pamf_channel_pl.publish_source import publish
from .plan import RUN,validate_runtime_context

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--repo',required=True);p.add_argument('--output',required=True)
    p.add_argument('--context',required=True);p.add_argument('--release-id',required=True);a=p.parse_args()
    validate_runtime_context(json.loads(Path(a.context).read_text(encoding='utf-8')))
    publish(a.repo,a.output,a.context,a.release_id,run_id=RUN,entrypoint='experiments.pamf_peer_supervised.runner',
        launch_owner='codex-pamf-peer-supervised-20261025',allowed_gpus=tuple(range(8)))
