import argparse,json
from pathlib import Path
from . import design as d
from experiments.cvs_report_ssl import publish as transport
from experiments.cvs_sinc_units50.publish import configure as previous
from experiments.cvs_sigtime_pair_seeds.publish import INSPECT as six_rows
INSPECT=six_rows.replace('experiments.cvs_sigtime_pair_seeds','experiments.cvs_sinc_units_seeds')
def configure():
 previous();transport.d=d
 transport.PATHS=list(dict.fromkeys([*transport.PATHS,'experiments/cvs_sigtime_pair_seeds','experiments/cvs_sinc_units_seeds']))
 transport.REMOTE=transport.REMOTE.replace('experiments.cvs_sinc_units50.preflight','experiments.cvs_sinc_units_seeds.preflight').replace('experiments.cvs_sinc_units50.dispatch','experiments.cvs_sinc_units_seeds.dispatch');transport.INSPECT=INSPECT
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--inspect',action='store_true');a=p.parse_args();configure()
 v=transport.inspect(a.output) if a.inspect else transport.publish(a.output)
 print(json.dumps(dict(status=v['status'],workers=len(v['workers']),epochs=[w['epoch_count'] for w in v['workers']],completion=v['completion'],failure=v['failure'])))
