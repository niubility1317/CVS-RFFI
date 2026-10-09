"""Read every completed structured training record and full source stdout."""
import json
from experiments.cvs_receiver_residual_test_recovery import design as d
from experiments.cvs_receiver_residual_v2.publish import ssh

SCRIPT=r'''
import csv,json,math,re
from pathlib import Path
p=Path(PROJECT);result=[]
def finite(x):
    if isinstance(x,float) and not math.isfinite(x):raise ValueError('Nonfinite logged number')
    if isinstance(x,dict):
        for v in x.values():finite(v)
    elif isinstance(x,list):
        for v in x:finite(v)
expected=json.loads((p/'runs/phase1_daot_rc4_pure_game_m3_20260917_r2/source_contract.json').read_text())
classes=['14-10','14-7','20-15','20-19','6-15','8-20']
for run in RUNS:
    for seed in (2026092701,2026092702,2026092703,2026092704):
        for arm in ('baseline','displacement','contribution','combined'):
            rid=arm+'-s'+str(seed);base=p/'runs'/run['parent_run_id']/rid/'source'
            done=json.loads((base/'completion.json').read_text());cfg=json.loads((base/'resolved_config.json').read_text())
            contract=json.loads((base/'source_contract.json').read_text())
            if any(contract.get(k)!=v for k,v in expected.items()) or contract['classes']!=classes:raise ValueError('Source contract differs')
            if done['epoch']!=200 or done['steps']!=10000 or done['target_access'] or done['target_evaluated']:raise ValueError('Source completion differs')
            steps=0;epochs=[];first_aux=None;first_sat=None
            with (base/'step_metrics.jsonl').open() as f:
                for line in f:
                    x=json.loads(line);finite(x);steps+=1
                    if x['step']!=steps or x['epoch']!=(steps-1)//50+1:raise ValueError('Step sequence differs')
                    if x['pseudo_labels_active'] or x['ema_active'] or x['domain_backbone_active']:raise ValueError('Forbidden method active')
                    if x['auxiliary_active'] and first_aux is None:first_aux=x['epoch']
                    if x['satellite_ce'] is not None and first_sat is None:first_sat=x['epoch']
            with (base/'epoch_metrics.jsonl').open() as f:
                for line in f:
                    x=json.loads(line);finite(x);epochs.append(x)
                    if x['epoch']!=len(epochs) or x['optimizer_steps']!=50 or x['source_sample_exposure']!=6300:raise ValueError('Epoch budget differs')
            with (base/'epoch_metrics.csv').open() as f:csvrows=list(csv.DictReader(f))
            if steps!=10000 or len(epochs)!=200 or len(csvrows)!=200 or first_sat!=80:raise ValueError('Log coverage differs')
            if first_aux!=(None if arm=='baseline' else 41):raise ValueError('Auxiliary activation differs')
            bank=[]
            with (base/'centroid_diagnostics.jsonl').open() as f:
                for line in f:
                    x=json.loads(line);finite(x);bank.append(x['epoch'])
            if bank!=list(range(40,201)):raise ValueError('Source statistics pass differs')
            log=p/'logs'/run['parent_run_id']/('source-'+rid+'.log');warnings=[];bad=[];lines=0
            with log.open(errors='replace') as f:
                for line in f:
                    lines+=1
                    if re.search(r'Traceback|RuntimeError:|out of memory|^Killed|Segmentation fault|FloatingPointError:',line):bad.append(line.strip())
                    if 'Warning' in line:warnings.append(line.strip())
            if bad:raise ValueError('Source stdout error '+repr(bad[:3]))
            result.append(dict(parent_run_id=run['parent_run_id'],row_id=rid,arm=arm,model_seed=seed,
                source_commit=cfg['commit'],steps_parsed=steps,epochs_parsed=len(epochs),csv_rows=len(csvrows),bank_epochs=len(bank),
                stdout_lines=lines,warnings=warnings,source_class_order=classes,physical_roles='EXACT_MATCH',
                first_auxiliary_epoch=first_aux,first_satellite_CE_epoch=first_sat,
                final_source_accuracy=epochs[-1]['source_val_accuracy'],final_source_worst_RX=epochs[-1]['source_val_worst_rx'],
                final_loss=epochs[-1]['total_loss'],parameters=cfg['total_parameters'],source_elapsed_seconds=done['elapsed_seconds']))
print(json.dumps(dict(status='VERIFIED',full_scan=True,rows=result,total_steps=sum(r['steps_parsed'] for r in result),total_epochs=sum(r['epochs_parsed'] for r in result),truth_read=False)))
'''


def main():
    value=json.loads(ssh(SCRIPT.replace('PROJECT',repr(d.PROJECT)).replace('RUNS',repr(d.RUNS))))
    output=d.ROOT/'local_artifacts'/d.RELEASE;output.mkdir(exist_ok=True)
    (output/'completed_training_audit.json').write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in value.items() if k!='rows'}))


if __name__=='__main__':main()
