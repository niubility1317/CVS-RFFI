"""Freeze new candidate and exact original baseline predictions before any scoring."""
from copy import deepcopy
from pathlib import Path
from experiments.cric_dura_comparison.io import read,write
from experiments.pamf_single_view_paper.plan import TASKS,READOUTS
from experiments.pamf_repl_view_shuffle import plan as tri
from .plan import RUN,candidate_rows,configuration,paper_row


def prepare(rt):
    if rt['run_id']!=RUN:raise ValueError('Registered extra-seed run required')
    root,project,ctx=Path(rt['run_root']),Path(rt['project']),read(rt['context'])
    h=ctx['handles']['F0'];jobs=[];sources={};all_predictions=[]
    for row in candidate_rows():
        seed=row['model_seed']
        for rid in (row['row_id'],f'tri-baseline-s{seed}'):
            all_predictions.append(str(root/'readouts'/rid/'predictions'/'complete.json'))
    def add(rid,kind,c,folder,status,dependencies=()):
        path=root/'configs'/(kind+'__'+rid+'.json');write(path,c)
        jobs.append(dict(id=kind+'__'+rid,kind=kind,config=str(path),completion=str(folder/'complete.json'),
                         successful_statuses=[status],dependencies=list(dependencies),nonblocking=False))
    for row in candidate_rows():
        seed=row['model_seed'];candidate=row['row_id'];baseline=f'tri-baseline-s{seed}'
        c=configuration(ctx,row,project,rt['code_commit']);c['frozen_sources']=rt['frozen_sources']
        source=Path(c['output_root']);add(candidate,'train',c,source,'SOURCE_FROZEN')
        original=project/'runs'/tri.RUN/'F'/f'main_split-repl-shuffled_time-3view-s{seed}'/'source'
        done=read(original/'complete.json');oc=done['config'];tri.validate_config(oc)
        if (done['status']!='SOURCE_FROZEN' or done['epoch']!=50 or done['updates']!=11100
            or done['target_access'] or done['U_truth_read'] or oc['model_seed']!=seed
            or read(oc['source_contract'])!=read(h['source_contract'])):
            raise ValueError('Original exact-seed F0 Tri E50 baseline required')
        for rid,folder,deps in ((candidate,source,[str(source/'complete.json')]),(baseline,original,[])):
            out=root/'readouts'/rid
            p=dict(run_id=RUN,row_id=rid,paper_row=paper_row(seed),protocol_id='F0',model_seed=seed,
                source_output=str(folder),source_contract=h['source_contract'],classes=h['classes'],source_days=[1,2,3],
                tests={t:dict(views_root=h['tests'][t]['views_root']) for t in TASKS},
                restored_caches={k:v['path'] for k,v in ctx['caches']['F0'].items()},views=row['views'],
                readouts=list(READOUTS),reuse_predictions=None,output_root=str(out/'predictions'),
                frozen_sources=rt['frozen_sources'])
            s=deepcopy(p);s.update(prediction_output=p['output_root'],output_root=str(out/'scores'),
                                   tests={t:h['tests'][t] for t in TASKS})
            for k in ('source_output','restored_caches','frozen_sources','reuse_predictions'):s.pop(k,None)
            add(rid,'predict',p,out/'predictions','ALL_PREDICTIONS_COMPLETE',deps)
            add(rid,'score',s,out/'scores','SCORED_COMPLETE',all_predictions)
            sources[rid]=dict(source_output=str(folder),predictions=p['output_root'],scores=s['output_root'],
                              reused=(rid==baseline))
    write(root/'jobs.json',jobs);write(root/'sources.json',sources)
    return jobs
