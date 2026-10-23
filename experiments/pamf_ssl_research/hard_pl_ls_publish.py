"""Pin the three-seed label-smoothing comparison to released no-U-As slots."""
import argparse
from copy import deepcopy
from .publish import publish
from .hard_pl_publish import prepare as original_prepare
from .hard_pl_ls import RUN,HARD_RUN,NO_UAS_RUN,METHOD,RESOURCE_GPU_BY_SEED,configuration


def prepare(context,project,release,commit):
    rt,original=original_prepare(context,project,release,commit)
    rt.update(run_id=RUN,run_root=project+'/runs/'+RUN,methods=[METHOD],max_total_workers=3,
        launch_owner='codex-pamf-hard-PL-LS-20261024',gpu_predecessor_state=project+'/runs/'+NO_UAS_RUN+'/state.json',
        prior_score_owner_state=project+'/runs/'+NO_UAS_RUN+'/state.json',hard_PL_GPU_owner_state=project+'/runs/'+HARD_RUN+'/state.json')
    rt.pop('udap_GPU_owner_state',None);rt.pop('uvb_GPU_owner_state',None)
    rows=[];blobs={}
    for old in rt['rows']:
        if old['method']!='HPL2':continue
        seed=old['model_seed'];c=configuration(context[str(seed)]['F21']['train'],project,commit)
        rid=c['row_id'];prefix=rt['run_root']+'/'+rid;stem='configs/'+old['row_id']
        p=deepcopy(original[stem+'-predict.json']);p.update(run_id=RUN,row_id=rid,source_run_id=RUN,source_row_id=rid,
            source_output=c['output_root'],output_root=prefix+'/predictions')
        s=deepcopy(original[stem+'-score.json']);s.update(run_id=RUN,row_id=rid,prediction_output=p['output_root'],output_root=prefix+'/scores')
        paths={key:release+'/configs/'+rid+'-'+key+'.json' for key in ('train','predict','score','row')}
        row=dict(old,row_id=rid,method=METHOD,train_config=paths['train'],prediction_config=paths['predict'],
            score_config=paths['score'],row_config=paths['row']);rows.append(row)
        for key,value in (('train',c),('predict',p),('score',s),('row',row)):blobs['configs/'+rid+'-'+key+'.json']=value
    rt['rows']=rows;rt['runtime_file']=release+'/runtime.json';blobs['runtime.json']=rt
    from .control import validate_rows
    validate_rows(rows,[METHOD],RESOURCE_GPU_BY_SEED)
    return rt,blobs


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for key in ('repo','context','output','ssh-config'):parser.add_argument('--'+key,required=True)
    a=parser.parse_args();publish(a.repo,a.context,a.output,a.ssh_config,prepare_run=prepare,run_id=RUN)
