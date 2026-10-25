"""Independent immutable low-label release; original nine workers are untouched."""
import argparse,json
from pathlib import Path
from experiments.pamf_channel_pl.publish_source import publish
from .plan import EXTRA_RUN,validate_runtime_context

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--repo',required=True);p.add_argument('--output',required=True)
    p.add_argument('--context',required=True);p.add_argument('--release-id',required=True);a=p.parse_args()
    validate_runtime_context(json.loads(Path(a.context).read_text(encoding='utf-8')))
    publish(a.repo,a.output,a.context,a.release_id,run_id=EXTRA_RUN,entrypoint='experiments.pamf_peer_supervised.extension_runner',
        launch_owner='codex-pamf-peer-label01-20261025',allowed_gpus=(5,6,7))
