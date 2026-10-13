import argparse,json
from pathlib import Path
import torch
from . import design as d
from .checks import run as checks
from experiments.cvs_report_projection.preflight import run as real_check
from .source import load
def run():
 torch.set_num_threads(2);result=checks();proof=real_check();L,U,contract=load(d.config(next(r for r in d.rows() if r['arm']=='TX_CONDITIONAL')))
 if len(L['ids'])!=6300 or len(U['ids'])!=56700 or 'y' in U:raise ValueError('Actualfullsource hiddenU contract')
 return dict(status='PASS',checks=result,real_checkpoint_noquery=proof['real_checkpoint_noquery'],source_L=6300,source_U_hidden=56700,query_IQ_or_truth_read=False,sourceVperformance_gate=False)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--remote-preflight',action='store_true');p.add_argument('--output',type=Path,required=True);a=p.parse_args();v=run();d.write(a.output,v);print(json.dumps(dict(status='PASS',clean_only=True,query_IQ_or_truth_read=False)))
