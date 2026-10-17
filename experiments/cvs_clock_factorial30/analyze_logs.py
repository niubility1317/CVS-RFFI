"""Read persisted E20+10 logs;missing transient counters remain explicitly missing."""
import argparse,json
from pathlib import Path
from experiments.cvs_clock_factorial20.analyze_logs import collect,read_json
ARMS=('B00','B10','B01','B11','FULL_AS')
def collect_e30(training_root,recovery_root,closed=False):
 training_root=Path(training_root);recovery_root=Path(recovery_root)
 value=collect(training_root,False,None,ARMS,10)
 value.update(schema='F_CLOCK_E20PLUS10_CLOSED_LOGS_V1',cumulative_epochs_each=30,new_epochs_each=10,recovery_run_id=recovery_root.name)
 for arm,row in value['rows'].items():
  base=recovery_root/(arm+'-s2026092701');source=training_root/(arm+'-s2026092701')/'source'
  if row['step_metrics']['records']!=2220 or row['step_metrics']['partial'] or not row['step_metrics']['step_order']['consecutive'] or row['epoch_file']['partial'] or [x['epoch'] for x in row['epoch_metrics']]!=list(range(1,11)):raise ValueError('Incomplete actual ten-epoch log:'+arm)
  row['EMA_initialization']=read_json(source/'EMA_initialization.json');row['optimizer_initialization']=read_json(source/'optimizer_initialization.json')
  row['recovery']=read_json(base/'source_recovery.json')
  if (row['recovery'] or {}).get('status')!='SOURCE_BUDGET_AND_FIXED_CHECKPOINT_VERIFIED':raise ValueError('Fixed saved-final source not verified:'+arm)
  row['missing_transient_records']=[name for name in ('reference_steps','unit_steps') if not row[name]['exists']]
  if row['training_sequence']['partial'] or row['training_sequence']['unexpected_roles']:raise ValueError('Invalid source physical sequence')
  if closed:
   row['prediction']=read_json(base/'prediction/complete.json');row['score']=read_json(base/'scores/metrics.json');row['row_completion']=read_json(base/'completion.json')
   if (row['row_completion'] or {}).get('status')!='TRAINED_DEFAULT_PREDICTED_SCORED':raise ValueError('Prediction/score incomplete:'+arm)
 if not all(x['same_complete_sequence'] for x in value['sampling_comparisons'].values()):raise ValueError('L/U physical sampling differs')
 if closed:
  value['matrix_completion']=read_json(recovery_root/'completion.json');value['result']=read_json(recovery_root/'result.json')
  if (value['matrix_completion'] or {}).get('status')!='ARTIFACTS_COMPLETE':raise ValueError('Matrix not closed')
 return value

def main():
 p=argparse.ArgumentParser();p.add_argument('training_root');p.add_argument('recovery_root');p.add_argument('--closed',action='store_true');a=p.parse_args()
 json.dump(collect_e30(a.training_root,a.recovery_root,a.closed),__import__('sys').stdout,ensure_ascii=False,allow_nan=False);print()
if __name__=='__main__':main()
