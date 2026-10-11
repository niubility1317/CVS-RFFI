"""Read existing N607 source artifacts, then compute fixed CPU-only diagnostics."""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path, PurePosixPath
import torch
from . import design as d


def inspect_source():
    from experiments.cvs_phase1_stack.publish import ssh
    payload = '''
import json
from pathlib import Path
base=Path(SOURCE)
def read(p): return json.loads(p.read_text())
rows=[]
for s in SEEDS:
 p=base/('E20-s'+str(s))
 rows.append(dict(seed=s,completion=read(p/'completion.json'),resolved=read(p/'resolved_config.json'),contract=read(p/'source_contract.json'),prediction_bytes=(p/'source_predictions.pt').stat().st_size))
print(json.dumps(dict(completion=read(base/'completion.json'),rows=rows)))
'''.replace('SOURCE', repr(d.SOURCE.as_posix())).replace('SEEDS', repr(list(d.SEEDS)))
    value = json.loads(ssh(payload))
    assert value['completion']['status'] == 'SOURCE_DIAGNOSTIC_COMPLETE'
    assert not value['completion']['target_access'] and value['completion']['identity_updates'] == 0
    for row in value['rows']:
        done, resolved = row['completion'], row['resolved']
        assert done['status'] == 'SOURCE_DIAGNOSTIC_COMPLETE'
        assert done['target_access'] is False and done['V_backward'] is False and done['U_truth'] is False
        assert resolved['model_seed'] == row['seed'] and resolved['source_only'] is True
        assert resolved['identity_updates'] == 0 and resolved['target_access'] is False
        assert resolved['V_backward'] is False and resolved['U_truth'] is False
        assert resolved['source_checkpoint'] == str(PurePosixPath(d.PROJECT)/'runs'/'20261011-phase1-ce-epoch10-20-30-40-manysig-m12-r01'/f"E20-s{row['seed']}"/'source'/'final_ssdg.pth')
        assert done['config'] == {k: resolved[k] for k in done['config']}
        provenance = resolved['parent_provenance']
        assert provenance['status'] == 'VERIFIED' and provenance['source_roles'] == 'EXACT_MATCH'
        assert provenance['seed'] == row['seed'] and provenance['epoch'] == 20
        assert provenance['checkpoint'] == resolved['source_checkpoint']
        assert provenance['scratch_only'] is True and provenance['ancestors'] == []
        assert provenance['training_target_access'] is False
        contract = row['contract']
        assert contract['native_role_comparison'] == 'EXACT_MATCH'
        assert contract['target_access_before_freeze'] is False
        assert contract['counts'] == dict(L_s=6300, U_s=56700, V=27000)
        if row['prediction_bytes'] <= 0:
            raise ValueError('Empty source predictions')
    return value


def download():
    if d.BASE.exists():
        raise FileExistsError('Existing local output; inspect it, do not overwrite or resubmit')
    evidence = inspect_source()
    d.BASE.mkdir(parents=True, exist_ok=False)
    d.write(d.BASE/'source_readback.json', evidence)
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=d.ROOT, text=True).strip()
    d.write(d.BASE/'resolved_config.json', dict(d.config(), execution_commit=commit,
            pid=os.getpid(), cwd=os.getcwd(), python=sys.executable, started_at=time.time()))
    for row in evidence['rows']:
        seed = row['seed']
        partial = d.BASE/f'P0-s{seed}.pt.part'
        dest = d.BASE/f'P0-s{seed}.pt'
        remote = str(d.SOURCE/f'E20-s{seed}'/'source_predictions.pt')
        subprocess.run(['scp', '-F', 'E:/type10-7/tools/n607_ssh_config',
                        '-o', 'BatchMode=yes', f'N607:{remote}', str(partial)], check=True)
        if partial.stat().st_size != row['prediction_bytes']:
            raise ValueError('Incomplete source transfer')
        partial.rename(dest)
        print(json.dumps(dict(event='SOURCE_TRANSFER_VERIFIED', seed=seed, bytes=dest.stat().st_size)), flush=True)
    d.write(d.BASE/'download_complete.json', dict(status='SOURCE_INPUTS_LOCAL', rows=3, target_access=False))


def analyze():
    from .audit import analyze as compute, select_p0
    if any((d.BASE/name).exists() for name in ('completion.json', 'analysis.json')):
        raise FileExistsError('Existing complete or partial result; preserve and inspect it')
    assert d.read(d.BASE/'download_complete.json')['status'] == 'SOURCE_INPUTS_LOCAL'
    with (d.BASE/'analysis_started.json').open('x', encoding='utf-8') as stream:
        json.dump(dict(owner=d.OWNER, pid=os.getpid(), started_at=time.time()), stream)
    started = time.perf_counter()
    torch.set_num_threads(2)
    data = {}
    contracts = {r['seed']: r['contract'] for r in d.read(d.BASE/'source_readback.json')['rows']}
    for seed in d.SEEDS:
        payload = torch.load(d.BASE/f'P0-s{seed}.pt', map_location='cpu', weights_only=True)
        data[seed] = select_p0(payload)
        del payload
        for role, count in d.config()['roles'].items():
            if len(data[seed][role]['ids']) != count:
                raise ValueError('Wrong registered source role count')
            key = 'L_s' if role == 'L' else 'V'
            if set(data[seed][role]['ids']) != set(contracts[seed]['role_ids'][key]):
                raise ValueError('Prediction physical IDs differ from actual source contract')
    with torch.no_grad():
        results = compute(data)
    d.write(d.BASE/'analysis.json', results)
    d.write(d.BASE/'completion.json', dict(status='SOURCE_DIAGNOSTIC_COMPLETE', input_run=d.SOURCE_RUN,
            source_P0_readouts=3*(6300+27000), new_model_forwards=0, model_updates=0, target_access=False,
            U_access=False, V_backward=False, seconds=time.perf_counter()-started, seeds=list(d.SEEDS),
            output=str(d.BASE/'analysis.json'), pid=os.getpid(), python=sys.executable))
    print(json.dumps(d.read(d.BASE/'completion.json')))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('mode', choices=['inspect', 'download', 'analyze'])
    args = p.parse_args()
    if args.mode == 'inspect':
        value = inspect_source()
        d.write(d.REPORT/'evidence/source_input_readback.json', value)
        print(json.dumps(dict(status='VERIFIED', rows=3, sizes=[r['prediction_bytes'] for r in value['rows']])))
    elif args.mode == 'download':
        download()
    else:
        analyze()
