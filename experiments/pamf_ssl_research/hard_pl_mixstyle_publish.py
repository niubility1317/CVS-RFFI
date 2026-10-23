"""Publish one fixed style candidate; share each seed card with healthy LS."""
import argparse
from .publish import publish
from .hard_pl_ls_publish import prepare as ls_prepare
from .hard_pl_mixstyle import RUN,LS_RUN,METHOD,RESOURCE_GPU_BY_SEED,configuration


def prepare(context,project,release,commit):
    rt,original=ls_prepare(context,project,release,commit)
    rt.update(run_id=RUN,run_root=project+'/runs/'+RUN,methods=[METHOD],
        launch_owner='codex-pamf-hard-PL-MixStyle-20261024',LS_GPU_owner_state=project+'/runs/'+LS_RUN+'/state.json',
        prior_score_owner_state=project+'/runs/'+LS_RUN+'/state.json')
    rows=[];blobs={}
    for old in rt['rows']:
        seed=old['model_seed'];c=configuration(context[str(seed)]['F21']['train'],project,commit)
        rid=c['row_id'];prefix=rt['run_root']+'/'+rid;stem='configs/'+old['row_id']
        p=dict(original[stem+'-predict.json']);p.update(run_id=RUN,row_id=rid,source_run_id=RUN,source_row_id=rid,
            source_output=c['output_root'],output_root=prefix+'/predictions')
        s=dict(original[stem+'-score.json']);s.update(run_id=RUN,row_id=rid,prediction_output=p['output_root'],output_root=prefix+'/scores')
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
