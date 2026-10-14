import argparse,json
from pathlib import Path
from . import design as d
from experiments.cvs_u_schedule50.publish import configure as parent
from experiments.cvs_report_ssl import publish as transport
def configure():
 parent();transport.d=d;transport.PATHS=list(dict.fromkeys([*transport.PATHS,'experiments/cvs_pa_recomposition']))
 transport.REMOTE=transport.REMOTE.replace('experiments.cvs_u_schedule50','experiments.cvs_pa_recomposition')
 transport.INSPECT=transport.INSPECT.replace('from experiments.cvs_u_schedule50 import design as d','from experiments.cvs_pa_recomposition import design as d')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--inspect',action='store_true');a=p.parse_args();configure();v=transport.inspect(a.output) if a.inspect else transport.publish(a.output);print(json.dumps(dict(status=v['status'],workers=len(v['workers']),completion=v['completion'],failure=v['failure'])))
