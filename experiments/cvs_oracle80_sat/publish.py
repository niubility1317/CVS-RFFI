import argparse,json
from pathlib import Path
from . import design as d
from experiments.cvs_report_ssl import publish as transport

INSPECT=r'''
import json,sys,time
from pathlib import Path
c=CONFIG;release=Path(c['project'])/'releases'/c['release'];base=Path(c['project'])/'runs'/c['run']
sys.path[:0]=[str(release),str(release/'code')]
from experiments.cvs_receiver_residual_capacity4.control import process
from experiments.cvs_multi_state_action.dispatch import capacity
def read(p):return json.loads(p.read_text()) if p.is_file() else None
submit=read(release/'submit.json');launch=read(base/'launch.json');done=read(base/'completion.json');failure=read(base/'failure.json')
actual=process(submit['pid']) if submit else None
worker=process(launch['pid']) if launch else None
valid=bool(actual and actual['cwd']==str(release) and actual['argv']==submit['argv'])
log=Path(launch['log']) if launch else None
epochs=base/'E30-s2026092701/source/epoch_metrics.jsonl'
last=json.loads(epochs.read_text().splitlines()[-1]) if epochs.is_file() and epochs.stat().st_size else None
if done:
 score=read(base/'scores/scoring_complete.json')
 valid=done['status']=='EXPERIMENT_COMPLETE' and done['trained_models']==done['tested_models']==1 and score['prediction_count']==1176000 and score['truth_last']
result=dict(status='FAILED' if failure else 'VERIFIED' if valid else 'UNKNOWN',submit=submit,process=actual,worker=worker,launch=launch,
 queue=read(base/'queue_state.json'),completion=done,failure=failure,preflight=read(release/'preflight.json'),
 resolved=read(base/'E30-s2026092701/source/resolved_config.json'),source_completion=read(base/'E30-s2026092701/source/completion.json'),
 effective_satellite_args={k:v for k,v in (read(base/'E30-s2026092701/source/resolved_native_args.json') or {}).items() if k in ('use_concat_sat_channel_aug','concat_sat_ce_only','concat_sat_fused_ce_only','sat_training_mode','lambda_sat_cls','lambda_sat_cons','use_sat_consistency','concat_sat_start_epoch','sat_cons_start_epoch','sat_view_seed','sat_train_scenarios','sat_view_schedule','use_unlabeled','use_ema_teacher','labeled_ratio','unlabeled_ratio','source_val_ratio')},
 initialization=read(base/'E30-s2026092701/source/initialization.json'),
 latest_epoch={k:v for k,v in (last or {}).items() if k in ('epoch','train/loss','time/epoch_sec','train/a1_ema_successful_updates')},
 log_bytes=log.stat().st_size if log and log.is_file() else None,log_tail=log.read_text(errors='replace').splitlines()[-8:] if log and log.is_file() else [],
 dispatcher_log_tail=(release/'dispatcher.stdout.log').read_text(errors='replace').splitlines()[-12:] if (release/'dispatcher.stdout.log').is_file() else [],
 capacity={str(g):dict(pids=sorted(s['pids']),free_mb=s['free_mb']) for g,s in capacity().items()},read_at=time.time())
print(json.dumps(result))
'''

def configure():
    transport.d=d;transport.PATHS=list(dict.fromkeys([*transport.PATHS,'experiments/cvs_ce_epoch2040','experiments/cvs_oracle80','experiments/cvs_oracle80_sat']))
    transport.REMOTE=transport.REMOTE.replace('experiments.cvs_report_ssl.control_checks','experiments.cvs_oracle80_sat.checks').replace('experiments.cvs_report_ssl.dispatch','experiments.cvs_oracle80_sat.dispatch')
    transport.INSPECT=INSPECT

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True,type=Path);p.add_argument('--inspect',action='store_true');a=p.parse_args();configure()
    result=transport.inspect(a.output) if a.inspect else transport.publish(a.output)
    print(json.dumps(dict(status=result['status'],completion=result.get('completion'),failure=result['failure'],stage=(result['queue'] or {}).get('stage'),epoch=result['latest_epoch'],log_bytes=result['log_bytes'])))
