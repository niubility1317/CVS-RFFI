"""Reuse fixed clean MoPC pilot rows in the full native task without recomputation."""
from pathlib import Path
import shutil
from .io import read, require


def copy_pilot_rows(job,rt,manifest,lookup,output):
    ref=job['reuse'];old=Path(ref['root']).resolve();prior=read(ref['job']);old_rt=read(ref['runtime'])
    require(job['profile']==prior['profile']=='mopc_hr' and
            job['seed']==prior['seed'] and job['capsule']==prior['capsule'] and
            prior['points']==[[5,20]] and Path(prior['output']).resolve()==old, 'Pilot task identity mismatch')
    for key in ('run_root','data_root','clean_root','source_contract','source_cache'):
        require(Path(rt[key]).resolve()==Path(old_rt[key]).resolve(), 'Pilot source or data binding mismatch')
    done=read(old/'complete.json');payload=read(old/'predictions.json')
    require(done.get('status')=='PREDICTIONS_COMPLETE' and done.get('truth_read') is False and
            done.get('query_fit') is False and done.get('rows')==2 and
            done.get('capsule_id')==manifest['capsule_id'] and done.get('model_seed')==job['seed'] and
            payload.get('capsule')==job['capsule'] and payload.get('capsule_id')==manifest['capsule_id'] and
            payload.get('profile')=='mopc_hr' and payload.get('model_seed')==job['seed'], 'Pilot completion mismatch')
    rows=payload['rows'];actual=[(r['stage'],r['k'],r['new_count']) for r in rows]
    require(len(rows)==2 and set(actual)=={('A0',5,0),('C',5,20)}, 'Pilot rows mismatch')
    checked=[]
    for row in rows:
        spec=lookup[row['k'],row['new_count']];name=row['prediction']
        require(row['arm']=='mopc_hr' and row['input_path']=='P0' and
                row['input_condition']=='clean_ground_pre_satellite_normalized' and
                row['split_id']==spec['split_id'] and row['registered_classes']==spec['registered_classes'] and
                Path(name).name==name and (old/name).is_file(), 'Pilot row binding mismatch')
        checked.append((row,name))
    result=[]
    for row,name in checked:
        dest=Path(output)/name
        with (old/name).open('rb') as source,dest.open('xb') as target:shutil.copyfileobj(source,target)
        result.append(dict(row,reused_from=str(old/name)))
    return result
