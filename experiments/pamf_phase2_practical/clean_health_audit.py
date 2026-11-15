"""Validate completed clean predictions and recorded fit terms without truth."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import json
import math
from pathlib import Path

from .clean_auto_score import validate_prediction_rows
from .clean_score import verify_all
from .io import atomic, read, require, write


def finite_values(value):
    if isinstance(value,float):
        return math.isfinite(value)
    if isinstance(value,dict):
        return all(finite_values(v) for v in value.values())
    if isinstance(value,list):
        return all(finite_values(v) for v in value)
    return True


def audit_entry(args):
    plan, entry = args
    result = dict(plan=plan, id=entry['id'], group=entry['group'], valid=False)
    try:
        item=verify_all({'jobs':[entry]},{'jobs':1})[0]
        job=item['job']; result.update(profile=job.get('profile','R42'), bridge=job.get('mopc_bridge','original'))
        for row in item['payload']['rows']:
            row['prediction']=str((item['folder']/row['prediction']).resolve())
        validate_prediction_rows(item)
        folder=item['folder']; profile=job.get('profile','R42'); group=entry['group']
        required=[]
        if group=='native_da' and profile!='tweak_m1':
            required.append(folder/'native_fit.jsonl')
            required.extend(folder/f"B-K{row['k']}-fit.json"
                            for row in item['payload']['rows'] if row['stage']=='B')
        elif group=='native_cil' and profile not in ('issl_fixed_v1',):
            required.append(folder/'native_fit.jsonl')
            required.extend(folder/f"C-K{row['k']}-N{row['new_count']}-fit.json"
                            for row in item['payload']['rows']
                            if row['stage']=='C' and not row.get('reused_from'))
        elif group=='hybrid' and profile!='tweak_m1':
            required.append(folder/'fit.jsonl')
        elif group=='native_cil' and profile=='issl_fixed_v1':
            required.extend(folder/f"C-K{row['k']}-N{row['new_count']}-fit.jsonl"
                            for row in item['payload']['rows'] if row['stage']=='C')
        for file in required:
            require(file.is_file(),f'Missing expected fit log: {file.name}')
        fit_records=0
        files=sorted(set(folder.glob('*fit.json')) | set(folder.glob('*fit.jsonl')))
        for file in files:
            with file.open(encoding='utf-8') as stream:
                records=(json.loads(line) for line in stream if line.strip()) if file.suffix=='.jsonl' else [json.load(stream)]
                for record in records:
                    require(finite_values(record),f'Nonfinite recorded fit term: {file.name}')
                    fit_records+=1
        result.update(valid=True,rows=len(item['payload']['rows']),fit_records=fit_records,
                      fit_files=len(files),fit_status='RECORDED' if files else 'NOT_RECORDED')
    except Exception as exc:
        result['error']=f'{type(exc).__name__}: {exc}'
    return result


def run(configs, output, workers):
    require(1 <= workers <= 4 and configs, 'Wrong audit inputs')
    root=Path(output).resolve(); require(not root.exists(), 'Audit output exists')
    jobs=[]
    for file in configs:
        config=read(file)
        for entry in config['jobs']:
            if Path(entry['completion']).is_file():
                jobs.append((Path(file).parent.name,entry))
    require(jobs, 'No completed jobs')
    root.mkdir(parents=True)
    write(root/'owner.json',dict(jobs=len(jobs),workers=workers,truth_read=False))
    counts=Counter(); errors=[]; fit_jobs=0; row_count=0
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for i,result in enumerate(pool.map(audit_entry,jobs,chunksize=1),1):
            label=(result['group'],result.get('profile','UNKNOWN'),result.get('bridge','UNKNOWN'))
            counts[label+('valid' if result['valid'] else 'invalid',)]+=1
            if result['valid']:
                row_count+=result['rows'];fit_jobs+=int(result['fit_records']>0)
            else:
                errors.append(result)
            if i%100==0:
                atomic(root/'state.json',dict(status='RUNNING',processed=i,total=len(jobs),
                       invalid=len(errors),truth_read=False))
    summary=dict(status='COMPLETE',jobs=len(jobs),rows=row_count,jobs_with_fit_records=fit_jobs,
        groups=[dict(group=k[0],profile=k[1],bridge=k[2],status=k[3],count=v) for k,v in sorted(counts.items())],
        invalid=errors,truth_read=False,scope='prediction_structure_and_recorded_fit_finiteness')
    write(root/'complete.json',summary)
    atomic(root/'state.json',dict(status='COMPLETE',processed=len(jobs),total=len(jobs),
           invalid=len(errors),truth_read=False))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',action='append',required=True);p.add_argument('--output',required=True)
    p.add_argument('--workers',type=int,default=4)
    a=p.parse_args();run(a.config,a.output,a.workers)


if __name__=='__main__':main()
