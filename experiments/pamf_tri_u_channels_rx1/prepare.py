from copy import deepcopy
from pathlib import Path
from experiments.cric_dura_comparison.io import read,write
from experiments.pamf_paper_completion.plan import RUN as PAPER_RUN
from experiments.pamf_single_view_paper.plan import TASKS,READOUTS
from .plan import RUN,SETTING,candidate_rows,configuration


def prepare(rt):
    if rt['run_id']!=RUN:raise ValueError('Registered RX1 run required')
    root,project,ctx=Path(rt['run_root']),Path(rt['project']),read(rt['context'])
    h=ctx['handles']['F1_rx_repartition_r01'];old=read(project/'runs'/PAPER_RUN/'sources.json')
    jobs,sources=[],{}
    predictions=[str(root/'readouts'/r['row_id']/'predictions'/'complete.json') for r in candidate_rows()]
    def add(rid,kind,c,folder,status,dependencies=()):
        path=root/'configs'/(kind+'__'+rid+'.json');write(path,c)
        jobs.append(dict(id=kind+'__'+rid,kind=kind,config=str(path),completion=str(folder/'complete.json'),
                         successful_statuses=[status],dependencies=list(dependencies),nonblocking=False))
    for row in candidate_rows():
        rid,seed,paper_row=row['row_id'],row['model_seed'],row['paper_row'];pid=paper_row['protocol_id']
        for method,order in (('ssl','permuted'),('pl','ordered')):
            old_id=f'{SETTING}-factorial-{method}-full-{order}-s{seed}';entry=old[old_id]
            done=read(Path(entry['source_output'])/'complete.json');cfg=done['config']
            pred=read(Path(entry['predictions'])/'complete.json');score=read(Path(entry['scores'])/'complete.json')
            if (done['status']!='SOURCE_FROZEN' or done['epoch']!=50 or done['updates']!=11100
                or done['target_access'] or done['U_truth_read'] or done['head_fit_updates']
                or cfg['model_seed']!=seed or cfg['protocol_id']!=pid
                or read(cfg['source_contract'])!=read(h['source_contract'])
                or pred['status']!='ALL_PREDICTIONS_COMPLETE' or pred['truth_read'] or pred['query_fit']
                or pred['views']!=row['views'] or pred['readouts']!=list(READOUTS)
                or score['status']!='SCORED_COMPLETE'):
                raise ValueError('RX1 matched comparison differs')
            if method=='ssl' and (entry['row']!=paper_row or cfg.get('paper_row')!=paper_row):
                raise ValueError('Original RX1 Tri model required')
            if method=='pl' and (cfg.get('paper_method')!='ordinary_PL' or cfg['repl_row']['method']!='ordinary_PL'):
                raise ValueError('Original RX1 ordinary PL required')
            sources[f'{method}-rx1-s{seed}']=dict(source_output=entry['source_output'],
                predictions=entry['predictions'],scores=entry['scores'],reused=True)
        c=configuration(ctx,row,project,rt['code_commit']);c['frozen_sources']=rt['frozen_sources']
        source,out=Path(c['output_root']),root/'readouts'/rid
        add(rid,'train',c,source,'SOURCE_FROZEN')
        p=dict(run_id=RUN,row_id=rid,paper_row=paper_row,protocol_id=pid,model_seed=seed,
            source_output=str(source),source_contract=h['source_contract'],classes=h['classes'],source_days=[1,2,3],
            tests={t:dict(views_root=h['tests'][t]['views_root']) for t in TASKS},
            restored_caches={k:v['path'] for k,v in ctx['caches'][pid].items()},views=row['views'],
            readouts=list(READOUTS),reuse_predictions=None,output_root=str(out/'predictions'),frozen_sources=rt['frozen_sources'])
        s=deepcopy(p);s.update(prediction_output=p['output_root'],output_root=str(out/'scores'),tests={t:h['tests'][t] for t in TASKS})
        for k in ('source_output','restored_caches','frozen_sources','reuse_predictions'):s.pop(k,None)
        add(rid,'predict',p,out/'predictions','ALL_PREDICTIONS_COMPLETE',[str(source/'complete.json')])
        add(rid,'score',s,out/'scores','SCORED_COMPLETE',predictions)
        sources[rid]=dict(source_output=str(source),predictions=p['output_root'],scores=s['output_root'],reused=False)
    write(root/'jobs.json',jobs);write(root/'sources.json',sources)
    return jobs
