"""User-requested early evaluation of the fixed 13 CORE90 rows; no training writes."""
import argparse
import importlib.util
import json
import os
import re
from pathlib import Path
import subprocess
import sys
import time

ROWS = [f'M{i:02}' for i in range(15) if i not in (9, 10)]

def early_admission_safe(project, run):
    """Original owner cannot launch prediction while both native rows are far from E200."""
    current = json.loads((run / 'pipeline_state.json').read_text())
    if current['status'] != 'TRAINING':
        return False
    for rid in ('M09', 'M10'):
        entry = current['rows'][rid]
        if entry['status'] != 'RUNNING' or (run / rid / 'final_ssdg.pth').exists():
            return False
        argv = Path(f"/proc/{entry['pid']}/cmdline").read_bytes().decode().split('\0')
        if str(run / rid) not in argv:
            return False
        log = (project / 'logs' / run.name / (rid + '.train.log')).read_text()
        epochs = re.findall(r'\[EPOCH-END\] E(\d+)/200', log)
        if not epochs or int(epochs[-1]) > 180:
            return False
    return True

def validate_source(run):
    state = json.loads((run / 'pipeline_state.json').read_text())
    roles = json.loads((run / 'source_contract.json').read_text())['role_ids']
    for rid in ROWS:
        folder = run / rid
        c = json.loads((folder / 'completion.json').read_text())
        init = json.loads((folder / 'initialization.json').read_text())
        if state['rows'][rid]['status'] != 'TRAINING_COMPLETE' or c['epochs'] != 200:
            raise ValueError('row is not frozen: ' + rid)
        if not init['scratch_only'] or init['target_contact'] or init['checkpoint_sources']:
            raise ValueError('invalid initialization: ' + rid)
        if json.loads((folder / 'source_contract.json').read_text())['role_ids'] != roles:
            raise ValueError('source roles differ: ' + rid)
        if not (folder / 'final_ssdg.pth').is_file():
            raise FileNotFoundError(rid)
    return state

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--project', type=Path, required=True)
    p.add_argument('--run-id', required=True)
    p.add_argument('--evaluation-id', required=True)
    a = p.parse_args()
    for value in (a.run_id, a.evaluation_id):
        if Path(value).name != value or value in ('.', '..'):
            raise ValueError('unsafe identifier')
    run = a.project / 'runs' / a.run_id
    prior = validate_source(run)
    release = Path(prior['release'])
    spec = importlib.util.spec_from_file_location('frozen_dispatch', release / 'code/scripts/dispatch_xuc15.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    output = a.project / 'runs' / a.evaluation_id
    logs = a.project / 'logs' / a.evaluation_id
    if output.exists() or logs.exists():
        raise FileExistsError('evaluation already exists; reconcile instead of restarting')
    output.mkdir(); logs.mkdir()
    state = dict(status='PREDICTING', pid=os.getpid(), source_run=a.run_id,
                 evaluation_id=a.evaluation_id, release=str(release), rows={}, created=time.time())
    def save():
        mod.write(output / 'evaluation_state.json', state)
    save()
    target_base = a.project / 'runs' / mod.TARGET_BASE
    active = {}
    pending = list(ROWS)
    try:
        while active or pending:
            for rid, job in list(active.items()):
                rc = job['process'].poll()
                if rc is None:
                    continue
                job['log'].close(); del active[rid]
                if rc:
                    raise RuntimeError(f'prediction failed {rid}: {rc}')
                pred = output / rid / 'predictions.json'
                if json.loads(pred.read_text())['record_count'] != 672000:
                    raise ValueError('incomplete prediction ' + rid)
                state['rows'][rid].update(status='PREDICTIONS_FIXED', prediction=str(pred), finished=time.time())
                save()
            if pending and len(active) < 4 and not early_admission_safe(a.project, run):
                raise RuntimeError('early admission window closed; preserve outputs and let original owner continue')
            gpu = mod.available_gpu(active) if pending and len(active) < 4 else None
            if gpu is not None:
                rid = pending.pop(0)
                cmd = [sys.executable, str(release / 'code/scripts/predict_phase1_truth_last.py'),
                       '--checkpoint', str(run / rid / 'final_ssdg.pth'), '--output-root', str(output / rid),
                       '--input-package', str(target_base / 'target_inputs'),
                       '--recipe', str(release / 'configs/core90_recipe_reference.json'),
                       '--run-id', a.run_id, '--row-id', rid, '--mode', 'predict',
                       '--batch-size', '256', '--num-workers', '0', '--device', 'cuda:0']
                h = (logs / (rid + '.predict.log')).open('x')
                child = subprocess.Popen(cmd, cwd=release, env=mod.env_for(gpu), stdin=subprocess.DEVNULL,
                                         stdout=h, stderr=subprocess.STDOUT)
                active[rid] = dict(process=child, log=h, gpu=gpu)
                state['rows'][rid] = dict(status='PREDICTING', pid=child.pid, gpu=gpu, argv=cmd, started=time.time())
                save()
            time.sleep(5)
        mod.write(output / 'predictions_complete.json', dict(rows=ROWS, record_count=672000*len(ROWS), created=time.time()))
        state['status'] = 'SCORING'; save()
        for rid in ROWS:
            cmd = [sys.executable, str(release / 'code/scripts/score_phase1_truth_last.py'),
                   '--predictions', state['rows'][rid]['prediction'],
                   '--truth', str(target_base / 'target_truth/truth_sidecar.json'),
                   '--output', str(output / rid / 'score.json')]
            with (logs / (rid + '.score.log')).open('x') as h:
                child = subprocess.Popen(cmd, cwd=release, env=mod.env_for(''), stdout=h, stderr=subprocess.STDOUT)
                state['stage_worker'] = dict(pid=child.pid, row=rid, argv=cmd); save()
                if child.wait():
                    raise RuntimeError('scoring failed ' + rid)
            state.pop('stage_worker')
            state['rows'][rid]['status'] = 'SCORED'; save()
        state.update(status='COMPLETE', finished=time.time()); save()
    except Exception as error:
        state.update(status='TECHNICAL_FAILURE', error=repr(error), failed=time.time()); save()
        # Preserve active predictors and their recorded PIDs; no blind retry.
        raise

if __name__ == '__main__':
    main()
