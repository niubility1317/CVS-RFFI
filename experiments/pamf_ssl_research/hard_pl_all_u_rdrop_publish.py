"""Use the GPU slot released by StepGuard;sole scoring follows StepNoCap."""
import argparse
from copy import deepcopy
from .publish import publish
from .hard_pl_step_no_cap_publish import prepare as base_prepare
from .hard_pl_step_no_cap import RUN as NC_RUN
from .hard_pl_all_u_rdrop import RUN,METHODS,RESOURCE_GPU_BY_SEED,configuration


def prepare(context,project,release,commit):
    rt,original=base_prepare(context,project,release,commit)
    rt.update(run_id=RUN,run_root=project+'/runs/'+RUN,methods=list(METHODS),max_total_workers=3,
        launch_owner='codex-pamf-all-u-rdrop-20261024',step_no_cap_GPU_owner_state=project+'/runs/'+NC_RUN+'/state.json',
        prior_score_owner_state=project+'/runs/'+NC_RUN+'/state.json')
    rows=[];blobs={}
    for old in rt['rows']:
        seed=old['model_seed'];c=configuration(context[str(seed)]['F21']['train'],project,commit);rid=c['row_id'];prefix=rt['run_root']+'/'+rid
        stem='configs/'+old['row_id'];p=deepcopy(original[stem+'-predict.json'])
        p.update(run_id=RUN,row_id=rid,source_run_id=RUN,source_row_id=rid,source_output=c['output_root'],output_root=prefix+'/predictions')
        s=deepcopy(original[stem+'-score.json']);s.update(run_id=RUN,row_id=rid,prediction_output=p['output_root'],output_root=prefix+'/scores')
        paths={key:release+'/configs/'+rid+'-'+key+'.json' for key in ('train','predict','score','row')}
        row=dict(old,row_id=rid,method=METHODS[0],train_config=paths['train'],prediction_config=paths['predict'],score_config=paths['score'],row_config=paths['row']);rows.append(row)
        for key,value in (('train',c),('predict',p),('score',s),('row',row)):blobs['configs/'+rid+'-'+key+'.json']=value
    rt['rows']=rows;rt['runtime_file']=release+'/runtime.json';blobs['runtime.json']=rt
    from .control import validate_rows
    validate_rows(rows,METHODS,RESOURCE_GPU_BY_SEED);return rt,blobs


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ('repo','context','output','ssh-config'):p.add_argument('--'+key,required=True)
    a=p.parse_args();publish(a.repo,a.context,a.output,a.ssh_config,prepare_run=prepare,run_id=RUN)
