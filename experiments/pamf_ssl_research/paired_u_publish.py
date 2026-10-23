"""Seed-pinned independent20 fork; each card follows its existing DURA GPU rows."""
import argparse
from copy import deepcopy
from .publish import prepare as scaffold_prepare, publish
from .paired_u import RUN, REFERENCE_RUN, RESOURCE_GPU_BY_SEED, configuration
from .plan import SEEDS


def prepare(context,project,release,commit):
    rt,scaffold=scaffold_prepare(context,project,release,commit)
    rt.update(run_id=RUN,run_root=project+'/runs/'+RUN,methods=['PAIREDU'],max_total_workers=3,
        launch_owner='codex-pamf-paired-U-20261023',gpu_by_seed=RESOURCE_GPU_BY_SEED,allowed_gpus=[5,6,7],
        gpu_predecessor_state=project+'/runs/'+REFERENCE_RUN+'/state.json',
        prior_score_owner_state=project+'/runs/'+REFERENCE_RUN+'/state.json')
    rows,blobs=[],{}
    g=context[str(SEEDS[0])]['F21']['predict']['readouts']['DURA-LU']
    from .dura_reference import DURA_RELATIVE
    if (g['kind']!='original_F0_LU' or g['roles']!='LU'
            or g['checkpoint']!=project+'/runs/'+DURA_RELATIVE
            or any(context[str(seed)][arm]['predict']['readouts']['DURA-LU']!=g
                   for seed in SEEDS for arm in ('F20','F21'))):
        raise ValueError('Same user-fixed DURA v1.0 for both matched baselines required')
    for seed in SEEDS:
        old_id='RXPSM-s'+str(seed)
        old=next(r for r in rt['rows'] if r['row_id']==old_id)
        c=configuration(context['_rxpsm_parents'][str(seed)],project,commit)
        c['parent_prefix_training_seconds']=context['_parent_timing'][str(seed)]
        rid=c['row_id'];prefix=rt['run_root']+'/'+rid
        cp=scaffold['configs/'+old_id+'-predict.json']
        cp.update(run_id=RUN,row_id=rid,source_run_id=RUN,source_row_id=rid,
            source_output=c['output_root'],output_root=prefix+'/predictions',
            readouts={'raw':{},'DURA-LU':deepcopy(g)})
        cs=scaffold['configs/'+old_id+'-score.json']
        cs.update(run_id=RUN,row_id=rid,prediction_output=cp['output_root'],output_root=prefix+'/scores',
                  readouts=['raw','DURA-LU'])
        paths={key:release+'/configs/'+rid+'-'+key+'.json' for key in ('train','predict','score','row')}
        row=dict(old,row_id=rid,method='PAIREDU',gpu=RESOURCE_GPU_BY_SEED[seed],train_config=paths['train'],
            prediction_config=paths['predict'],score_config=paths['score'],row_config=paths['row'])
        rows.append(row)
        for key,value in (('train',c),('predict',cp),('score',cs),('row',row)):
            blobs['configs/'+rid+'-'+key+'.json']=value
    rt['rows']=rows
    rt['runtime_file']=release+'/runtime.json'
    blobs['runtime.json']=rt
    from .control import validate_rows
    validate_rows(rows,rt['methods'],RESOURCE_GPU_BY_SEED)
    return rt,blobs


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ('repo','context','output','ssh-config'):p.add_argument('--'+key,required=True)
    a=p.parse_args()
    publish(a.repo,a.context,a.output,a.ssh_config,prepare_run=prepare,run_id=RUN)
