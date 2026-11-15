"""Reuse fixed native MoPC source predictions when only C training changes."""
from pathlib import Path
import shutil

from .io import read,require


def copy_a0(job,rt,manifest,lookup,output):
    ref=job['reuse_a0'];old=read(ref['job']);old_rt=read(ref['runtime'])
    require(job.get('profile')==old.get('profile')=='mopc_hr' and
            all(job.get(k)==old.get(k) for k in ('capsule','seed','points')),
            'A0 source task identity differs')
    for field in ('run_root','data_root','clean_root','source_contract','source_cache','phase1_release'):
        require(Path(rt[field]).resolve()==Path(old_rt[field]).resolve(),'A0 source or data differs')
    folder=Path(old['output']).resolve();done=read(folder/'complete.json');index=read(folder/'predictions.json')
    require(done.get('status')=='PREDICTIONS_COMPLETE' and done.get('truth_read') is False and
            done.get('query_fit') is False and done.get('capsule_id')==manifest['capsule_id'] and
            index.get('capsule')==job['capsule'] and index.get('profile')=='mopc_hr' and index.get('model_seed')==job['seed'],
            'Original A0 completion differs')
    selected=[r for r in index['rows'] if r['stage']=='A0']
    expected={int(k) for k,n in job['points']}
    require(len(selected)==len(expected) and {r['k'] for r in selected}==expected,'Original A0 matrix incomplete')
    result=[]
    for row in selected:
        spec=lookup[row['k'],0];name=row['prediction']
        require(row['arm']=='mopc_hr' and row['input_path']=='P0' and row['split_id']==spec['split_id'] and
                row['registered_classes']==spec['registered_classes'] and Path(name).name==name and
                row['input_condition']=='clean_ground_pre_satellite_normalized' and (folder/name).is_file(),
                'Original A0 row differs')
    for row in selected:
        name=row['prediction']
        with (folder/name).open('rb') as src,(Path(output)/name).open('xb') as dst:shutil.copyfileobj(src,dst)
        result.append(dict(row,reused_from=str(folder/name)))
    return result
