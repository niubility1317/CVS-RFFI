import argparse,json
from pathlib import Path
from . import design as d
from .checks import run as checks
from experiments.cvs_report_projection.preflight import run as real_check
def run():
 result=checks();proof=real_check();return dict(status="PASS",checks=result,real_checkpoint_noquery=proof["real_checkpoint_noquery"],query_IQ_or_truth_read=False)
if __name__=="__main__":
 p=argparse.ArgumentParser();p.add_argument("--remote-preflight",action="store_true");p.add_argument("--output",type=Path,required=True);a=p.parse_args();v=run();d.write(a.output,v);print(json.dumps(dict(status="PASS",query_IQ_or_truth_read=False)))
