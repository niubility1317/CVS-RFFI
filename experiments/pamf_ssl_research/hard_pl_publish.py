"""Two ordinary-PL source forks per seed, never more than two own workers/card."""
import argparse
from copy import deepcopy
from .publish import prepare as scaffold_prepare,publish
from .hard_pl import RUN,METHODS,SEEDS,RESOURCE_GPU_BY_SEED,configuration
from .u_dualanchor import RUN as UDAP_RUN
from .u_viewbalance import RUN as UVB_RUN


def prepare(context,project,release,commit):
    rt,scaffold=scaffold_prepare(context,project,release,commit)
    rt.update(run_id=RUN,run_root=project+'/runs/'+RUN,methods=list(METHODS),max_total_workers=6,
        launch_owner='codex-pamf-hard-PL-T1U-20261024',gpu_by_seed=RESOURCE_GPU_BY_SEED,allowed_gpus=[5,6,7],
        udap_GPU_owner_state=project+'/runs/'+UDAP_RUN+'/state.json',
        gpu_predecessor_state=project+'/runs/'+UVB_RUN+'/state.json',
        prior_score_owner_state=project+'/runs/'+UVB_RUN+'/state.json',max_compute_processes_per_gpu=3)
    g=context[str(SEEDS[0])]['F21']['predict']['readouts']['DURA-LU']
    from .dura_reference import DURA_RELATIVE
    if (g['kind']!='original_F0_LU' or g['roles']!='LU' or g['checkpoint']!=project+'/runs/'+DURA_RELATIVE
        or any(context[str(seed)][arm]['predict']['readouts']['DURA-LU']!=g for seed in SEEDS for arm in ('F20','F21'))):
        raise ValueError('Same fixed DURA and matched baseline readouts required')
    rows=[];blobs={}
    for seed in SEEDS:
        old_id='RXPSM-s'+str(seed);old=next(r for r in rt['rows'] if r['row_id']==old_id)
        for method in METHODS:
            c=configuration(context[str(seed)]['F21']['train'],method,project,commit)
            rid=c['row_id'];prefix=rt['run_root']+'/'+rid
            p=deepcopy(scaffold['configs/'+old_id+'-predict.json'])
            p.update(run_id=RUN,row_id=rid,source_run_id=RUN,source_row_id=rid,source_output=c['output_root'],
                output_root=prefix+'/predictions',readouts={'raw':{},'DURA-LU':deepcopy(g)})
            s=deepcopy(scaffold['configs/'+old_id+'-score.json'])
            s.update(run_id=RUN,row_id=rid,prediction_output=p['output_root'],output_root=prefix+'/scores',readouts=['raw','DURA-LU'])
            paths={key:release+'/configs/'+rid+'-'+key+'.json' for key in ('train','predict','score','row')}
            row=dict(old,row_id=rid,method=method,gpu=RESOURCE_GPU_BY_SEED[seed],train_config=paths['train'],
                prediction_config=paths['predict'],score_config=paths['score'],row_config=paths['row'])
            rows.append(row)
            for key,value in (('train',c),('predict',p),('score',s),('row',row)):blobs['configs/'+rid+'-'+key+'.json']=value
    rt['rows']=rows;rt['runtime_file']=release+'/runtime.json';blobs['runtime.json']=rt
    from .control import validate_rows
    validate_rows(rows,METHODS,RESOURCE_GPU_BY_SEED)
    return rt,blobs


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for key in ('repo','context','output','ssh-config'):parser.add_argument('--'+key,required=True)
    args=parser.parse_args();publish(args.repo,args.context,args.output,args.ssh_config,prepare_run=prepare,run_id=RUN)
