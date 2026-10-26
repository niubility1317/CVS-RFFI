import argparse
import json
from pathlib import Path
from .receiver_probe import RUN,RELEASE,CACHE_RUN,RECIPE
from .publish_vat_source import publish


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--repo',required=True);p.add_argument('--output',required=True);p.add_argument('--context',required=True)
    a=p.parse_args();ctx=json.loads(Path(a.context).read_text(encoding='utf-8'))
    publish(a.repo,a.output,a.context,run_id=RUN,release_id=RELEASE,entrypoint='experiments.pamf_u_history.receiver_probe',
        required_relative=['runs/'+CACHE_RUN+'/complete.json'],runtime_extra=dict(recipe=RECIPE,G=ctx['G']))
