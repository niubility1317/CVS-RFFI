from copy import deepcopy
from pathlib import Path
from experiments.cric_dura_comparison.io import read,write
from experiments.pamf_single_view_paper.plan import TASKS,READOUTS
from experiments.pamf_tri_u_extra.plan import RUN as BASELINE_RUN,paper_row
from .plan import RUN,candidate_rows,configuration


def prepare(rt):
    if rt['run_id']!=RUN:raise ValueError('Registered phase extra run required')
    root,project,ctx=Path(rt['run_root']),Path(rt['project']),read(rt['context'])
    h=ctx['handles']['F0'];baseline_root=project/'runs'/BASELINE_RUN;old=read(baseline_root/'sources.json')
    jobs=[];sources={};predictions=[str(root/'readouts'/r['row_id']/'predictions'/'complete.json') for r in candidate_rows()]
    def add(rid,kind,c,folder,status,dependencies=()):
        path=root/'configs'/(kind+'__'+rid+'.json');write(path,c)
        jobs.append(dict(id=kind+'__'+rid,kind=kind,config=str(path),completion=str(folder/'complete.json'),
                         successful_statuses=[status],dependencies=list(dependencies),nonblocking=False))
    for row in candidate_rows():
        rid,seed=row['row_id'],row['model_seed'];brid=f'tri-baseline-s{seed}';entry=old[brid]
        prior_cfg=read(baseline_root/'configs'/f'predict__{brid}.json')
        marker=read(Path(entry['predictions'])/'complete.json');score=read(Path(entry['scores'])/'complete.json')
        expected=project/'runs/20261027-phase1-pamf-repl-shuffled-views-m16-r02/F'/f'main_split-repl-shuffled_time-3view-s{seed}'/'source'
        if (Path(entry['source_output'])!=expected or prior_cfg['model_seed']!=seed
            or marker['status']!='ALL_PREDICTIONS_COMPLETE' or marker['truth_read'] or marker['query_fit']
            or marker['run_id']!=BASELINE_RUN or marker['row_id']!=brid or marker['views']!=row['views']
            or marker['readouts']!=list(READOUTS)
            or score['status']!='SCORED_COMPLETE' or prior_cfg['restored_caches']!={k:v['path'] for k,v in ctx['caches']['F0'].items()}
            or prior_cfg['classes']!=h['classes'] or any(prior_cfg['tests'][t]['views_root']!=h['tests'][t]['views_root'] for t in TASKS)):
            raise ValueError('Exact same-seed original F0 baseline and physical views required')
        sources[brid]=dict(predictions=entry['predictions'],scores=entry['scores'],source_output=entry['source_output'],reused=True)
        c=configuration(ctx,row,project,rt['code_commit']);c['frozen_sources']=rt['frozen_sources']
        source=Path(c['output_root']);out=root/'readouts'/rid;add(rid,'train',c,source,'SOURCE_FROZEN')
        p=dict(run_id=RUN,row_id=rid,paper_row=paper_row(seed),protocol_id='F0',model_seed=seed,
            source_output=str(source),source_contract=h['source_contract'],classes=h['classes'],source_days=[1,2,3],
            tests={t:dict(views_root=h['tests'][t]['views_root']) for t in TASKS},
            restored_caches={k:v['path'] for k,v in ctx['caches']['F0'].items()},views=row['views'],
            readouts=list(READOUTS),reuse_predictions=None,output_root=str(out/'predictions'),frozen_sources=rt['frozen_sources'])
        s=deepcopy(p);s.update(prediction_output=p['output_root'],output_root=str(out/'scores'),tests={t:h['tests'][t] for t in TASKS})
        for k in ('source_output','restored_caches','frozen_sources','reuse_predictions'):s.pop(k,None)
        add(rid,'predict',p,out/'predictions','ALL_PREDICTIONS_COMPLETE',[str(source/'complete.json')])
        add(rid,'score',s,out/'scores','SCORED_COMPLETE',predictions)
        sources[rid]=dict(source_output=str(source),predictions=p['output_root'],scores=s['output_root'],reused=False)
    write(root/'jobs.json',jobs);write(root/'sources.json',sources)
    return jobs
