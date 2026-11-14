"""Independent truth-last scoring for fixed clean comparison predictions."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from .clean_scheduler import completion_matches
from .episodic_io import query_complement
from .episodic_score import metrics
from .io import read,require,write


def expected_rows(group,job):
    points={tuple(map(int,p)) for p in job['points']};ks={k for k,n in points};profile=job.get('profile')
    result=set()
    if group=='pamf':
        for k in ks:
            result.update(('R42','A0',k,0,f'Q{p}') for p in (0,1))
            result.update(('R42','B',k,0,f'S{p}Q{p}') for p in (0,1))
        for k,n in points:result.update(('R42','C',k,n,f'S{p}Q{p}') for p in (0,1))
    elif group=='native_da':
        path='S0Q0' if profile=='tweak_m1' else 'P0'
        for k in ks:result.update((profile,stage,k,0,path) for stage in ('A0','B'))
    elif group=='native_cil':
        path='S0Q0' if profile=='issl_fixed_v1' else 'P0'
        result.update((profile,'A0',k,0,path) for k in ks)
        result.update((profile,'C',k,n,path) for k,n in points)
    elif group=='hybrid':
        path='S0Q0' if profile=='tweak_m1' else 'P0'
        result.update((profile+'__'+method,'C',k,n,path) for k,n in points for method in job['methods'])
    else:raise ValueError('Unknown comparison group')
    return result


def key(row):
    return row['arm'],row['stage'],int(row['k']),int(row['new_count']),row['input_path']


def verify_all(config,preparation):
    """Check every requested prediction before any truth file is opened."""
    require(not preparation.get('blocked'),'Retained comparison methods remain blocked')
    jobs=config['jobs'];require(jobs and len(jobs)==preparation['jobs'],'Incomplete preregistered jobs')
    verified=[]
    for entry in jobs:
        require(completion_matches(entry),'A requested prediction job is incomplete')
        job=read(entry['job']);rt=read(entry['runtime']);folder=Path(entry['completion']).parent
        done=read(folder/'complete.json');payload=read(folder/'predictions.json');rows=payload['rows']
        if 'issl_ssl_lr' in job:
            require(done.get('issl_ssl_lr')==job['issl_ssl_lr'] and payload.get('issl_ssl_lr')==job['issl_ssl_lr']
                    and done.get('issl_budget')==job['issl_budget'] and payload.get('issl_budget')==job['issl_budget'],
                    'ISSL learning rate or budget does not match prediction')
        require(payload['capsule']==job['capsule'] and payload['model_seed']==job['seed'] and len(rows)==done['rows'],
                'Prediction index identity mismatch')
        actual=[key(row) for row in rows]
        require(len(actual)==len(set(actual)) and set(actual)==expected_rows(entry['group'],job),'Missing or unexpected method/stage/path/K/N row')
        require(all(row.get('input_condition')=='clean_ground_pre_satellite_normalized' for row in rows),
                'Non-clean input in clean comparison')
        require(all((folder/row['prediction']).is_file() for row in rows),'Missing prediction file')
        verified.append(dict(entry=entry,job=job,rt=rt,folder=folder,payload=payload))
    return verified


def score_job(item,output):
    job,rt,folder,payload=item['job'],item['rt'],item['folder'],item['payload']
    cap=Path(rt['data_root'])/job['capsule'];manifest=read(cap/'manifest.json')
    specs={s['split_id']:s for s in (read(p) for p in (cap/'splits').glob('*.json'))}
    pool,episode,rx=manifest['physical_pool'],manifest['episode_id'],manifest['receiver']
    # The original scene label belongs to the simulated satellite view; it is
    # not a physical clean-input condition and is deliberately not read.
    with np.load(Path(rt['data_root'])/'score_only'/pool/episode/(rx+'.npz'),allow_pickle=False) as z:
        truth=z['labels'];days=z['days']
    prepared=[]
    for row in payload['rows']:
        spec=specs[row['split_id']]
        require((spec['k'],spec['new_count'])==(row['k'],row['new_count']) and
                spec['registered_classes']==row['registered_classes'],'Wrong split/K/N/classes')
        registered=np.load(cap/'stage_rows'/f"N{spec['new_count']:02d}.npy",allow_pickle=False)
        expected=query_complement(registered,np.asarray(spec['support_indices']))
        with np.load(folder/row['prediction'],allow_pickle=False) as f:
            q=f['query_indices'];pred=f['predicted_indices']
            restricted=f['old_restricted_indices'] if 'old_restricted_indices' in f else None
        require(np.array_equal(q,expected) and len(q)==spec['query_count'] and len(pred)==len(q),'Wrong query rows')
        labels=truth[q];count=len(spec['registered_classes'])
        value=dict(row,physical_pool=pool,episode=episode,receiver=rx,model_seed=job['seed'],
                   metrics=metrics(labels,pred,count),
                   by_day={str(day):metrics(labels[days[q]==day],pred[days[q]==day],count)
                           for day in np.unique(days[q])})
        if restricted is not None:
            old=labels<6;value['old_restricted_macro']=metrics(labels[old],restricted[old],6)['macro_accuracy']
        prepared.append((row,q,pred,labels,value))
    for row,q,pred,labels,value in prepared:
        peers=[p for p in prepared if p[0]['arm']==row['arm'] and p[0]['stage']==row['stage'] and
               p[0]['new_count']==row['new_count'] and p[0]['input_path']==row['input_path']]
        if {p[0]['k'] for p in peers}!={1,5,10,20}:
            value['common_K_status']='UNAVAILABLE_INCOMPLETE_K_SET';value['common_K_metrics']=None
            continue
        common=peers[0][1]
        for p in peers[1:]:common=np.intersect1d(common,p[1],assume_unique=True)
        mask=np.isin(q,common);require(mask.sum()==len(common) and len(common)>0,'Invalid common-K query')
        value['common_K_status']='AVAILABLE';value['common_K_metrics']=metrics(labels[mask],pred[mask],len(row['registered_classes']))
    output.mkdir(parents=True,exist_ok=False)
    write(output/'metrics.json',dict(capsule_id=manifest['capsule_id'],model_seed=job['seed'],
          group=item['entry']['group'],rows=[p[-1] for p in prepared],truth_last=True,target_feedback=False,
          input_condition='clean_ground_pre_satellite_normalized',satellite_scene_labels_used=False))
    write(output/'complete.json',dict(status='SCORED_COMPLETE',rows=len(prepared),truth_last=True,target_feedback=False))


def run(config,preparation,output_root,workers=1):
    require(isinstance(workers,int) and 1<=workers<=4,'Scoring workers must be between 1 and 4')
    verified=verify_all(config,preparation)
    root=Path(output_root).resolve();run_root=Path(config['run_root']).resolve()
    require(root.is_relative_to(run_root) and root!=run_root and not root.exists(),'Unsafe score output')
    root.mkdir(parents=True,exist_ok=False)
    if workers==1:
        for item in verified:score_job(item,root/item['entry']['id'])
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures=[pool.submit(score_job,item,root/item['entry']['id']) for item in verified]
            for future in futures:future.result()
    write(root/'complete.json',dict(status='SCORED_COMPLETE',jobs=len(verified),truth_last=True,target_feedback=False))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True)
    p.add_argument('--preparation',required=True);p.add_argument('--output',required=True)
    p.add_argument('--workers',type=int,default=1,help='CPU scoring processes, at most 4')
    a=p.parse_args();run(read(a.config),read(a.preparation),a.output,a.workers)


if __name__=='__main__':main()
