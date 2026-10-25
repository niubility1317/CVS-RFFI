"""Split source/cache/prediction/truth configs and freeze the nine-row jobs."""
from copy import deepcopy
from pathlib import Path
from experiments.cric_dura_comparison.io import read,write
from experiments.practical_full_2x2.plan import TASKS
from experiments.pamf_channel_pl.references import resolve_reference_score_paths
from experiments.pamf_channel_pl.plan import validate_context
from .plan import RUN,rows,make_config,source_decision


def prepare(rt):
    context=read(rt['context']);validate_context(context)
    weight=source_decision(read(Path(context['restored_dose_root'])/'complete.json'))
    if weight!=context['restored_selected_weight']:raise ValueError('Source-frozen published dose differs')
    root=Path(rt['run_root']);cache=root/'source_U_cache';jobs=[];reference=context['reference'];refs={}
    wanted={'F0-'+m+'-s'+str(r['model_seed']) for r in rows() for m in ('supervised','ordinary_PL','REPL')}
    refpaths=resolve_reference_score_paths(context['reference_score_jobs'],wanted)
    def add(key,kind,c,completion,status,deps=(),pool=(0,),row_id=None):
        path=root/'configs'/(key+'.json');write(path,c)
        jobs.append(dict(id=key,kind=kind,row_id=row_id,config=str(path),completion=str(completion),
            successful_statuses=[status],dependencies=list(deps),gpu_pool=list(pool),
            execution_release=rt['release_root'],entrypoint='experiments.pamf_restored_pl.runner',execution_python=rt['python']))
    add('source_cache','cache',dict(source_context=context['restored_source_context'],output_root=str(cache),
        frozen_sources=rt['frozen_sources'],run_id=RUN,code_commit=rt['code_commit']),cache/'complete.json','SOURCE_U_CACHE_COMPLETE')
    configs={}
    for row in rows():
        cfg=make_config(context,row,rt['project'],rt['code_commit'],weight,str(cache));cfg['frozen_sources']=rt['frozen_sources']
        configs[row['row_id']]=cfg
    smoke_cfg=next(v for v in configs.values() if v['paper_method']=='DURASoft' and v['model_seed']==2026092701)
    smoke_train=root/'configs/smoke_train.json';write(smoke_train,smoke_cfg)
    add('source_backward_smoke','smoke',dict(train_config=str(smoke_train),output=str(root/'smoke.json'),frozen_sources=rt['frozen_sources']),
        root/'smoke.json','PASS',[str(cache/'complete.json')])
    for row in rows():
        c=configs[row['row_id']];rid=row['row_id'];s=row['model_seed'];h=reference['handles']['F0'];out=root/'readouts'/rid
        if read(c['source_contract'])!=read(h['source_contract']):raise ValueError('Same source/evaluation contract required')
        add('train__'+rid,'train',c,Path(c['output_root'])/'complete.json','SOURCE_FROZEN',
            [str(cache/'complete.json'),str(root/'smoke.json')],row['gpu_pool'],rid)
        pred=dict(row_id=rid,model_seed=s,protocol_id='F0',run_id=RUN,method=row['method'],source_output=c['output_root'],
            source_contract=h['source_contract'],classes=h['classes'],source_days=[1,2,3],
            tests={t:dict(views_root=h['tests'][t]['views_root']) for t in TASKS},
            restored_caches=dict(LU=reference['caches']['F0']['LU']['path']),readouts=['raw','DURA-LU'],
            output_root=str(out/'predictions'),frozen_sources=rt['frozen_sources'])
        add('predict__'+rid,'predict',pred,out/'predictions/complete.json','ALL_PREDICTIONS_COMPLETE',
            [str(Path(c['output_root'])/'complete.json')],row['gpu_pool'],rid)
        score=deepcopy(pred);score.update(prediction_output=pred['output_root'],output_root=str(out/'scores'),tests=h['tests'])
        for key in ('source_output','source_contract','restored_caches','frozen_sources'):score.pop(key)
        add('score__'+rid,'score',score,out/'scores/complete.json','SCORED_COMPLETE',
            [str(out/'predictions/complete.json')],(),rid)
        refs[rid]={m:refpaths['F0-'+m+'-s'+str(s)] for m in ('supervised','ordinary_PL','REPL')}
    write(root/'jobs.json',jobs);write(root/'reference_scores.json',refs)
    write(root/'frozen_scientific_matrix.json',dict(run_id=RUN,rows=rows(),weight=weight,postG_noise=False,
        source_decision=context['restored_dose_root'],target_feedback=False))
    return jobs
