"""Source configurations exclude target handles; independent scoring sees only predictions."""
from copy import deepcopy
from pathlib import Path
from experiments.cric_dura_comparison.io import read,write
from experiments.practical_full_2x2.plan import TASKS
from experiments.pamf_channel_pl.references import resolve_reference_score_paths
from .plan import RUN,rows,make_config


def source_admitted(proof):
    if (proof['status']!='SOURCE_CONSENSUS_COMPLETE' or proof['model_updates'] or proof['target_reads'] or proof['U_truth_read']):
        raise ValueError('Completed no-update source diagnostic required')
    tested={p['protocol']:p for p in proof['protocols']}
    for pid in ('F0','label01'):
        r=tested[pid]
        if r['source_V_seen']!=27000 or r['model_seeds']!=[2026092701,2026092702,2026092703]:
            raise ValueError('Complete original source consensus evidence required')
        v=next(v for v in r['results'] if v['policy']=='unanimous3' and v['dimension']=='overall')
        if v['precision_percent']<99.5 or v['coverage_percent']<85.:raise ValueError('Source consensus quality/coverage falsification failed')


def prepare(rt):
    context=read(rt['context']);source_admitted(read(Path(context['consensus_source_root'])/'complete.json'))
    reference=context['reference'];root=Path(rt['run_root']);jobs=[];sources={};refs={}
    wanted={'F0-'+m+'-s'+str(s) for r in rows() for s in (r['model_seed'],) for m in ('supervised','ordinary_PL','REPL')}
    reference_paths=resolve_reference_score_paths(context['reference_score_jobs'],wanted)
    def add(kind,row,c,completion,status,deps=()):
        path=root/'configs'/(kind+'__'+row['row_id']+'.json');write(path,c)
        jobs.append(dict(id=kind+'__'+row['row_id'],kind=kind,row_id=row['row_id'],model_seed=row['model_seed'],gpu=row['gpu'],
            config=str(path),completion=str(completion),successful_statuses=[status],dependencies=list(deps),
            execution_release=rt['release_root'],entrypoint='experiments.pamf_peer_supervised.runner',execution_python=rt['python'],lane='new'))
    for row in rows():
        c=make_config(context,row,rt['project'],rt['code_commit']);c['frozen_sources']=rt['frozen_sources']
        rid=row['row_id'];s=row['model_seed'];h=reference['handles']['F0']
        if read(c['source_contract'])!=read(h['source_contract']):raise ValueError('Source/evaluation contract differs')
        add('train',row,c,Path(c['output_root'])/'complete.json','SOURCE_FROZEN')
        sources[rid]=c['output_root'];out=root/'readouts'/rid
        pred=dict(row_id=rid,model_seed=s,protocol_id='F0',run_id=RUN,method=row['method'],
            source_output=c['output_root'],source_contract=h['source_contract'],classes=h['classes'],source_days=[1,2,3],
            tests={t:dict(views_root=h['tests'][t]['views_root']) for t in TASKS},
            restored_caches=dict(LU=reference['caches']['F0']['LU']['path']),readouts=['raw','DURA-LU'],
            output_root=str(out/'predictions'),frozen_sources=rt['frozen_sources'])
        add('predict',row,pred,out/'predictions/complete.json','ALL_PREDICTIONS_COMPLETE',[str(Path(c['output_root'])/'complete.json')])
        scorer=deepcopy(pred);scorer.update(prediction_output=pred['output_root'],output_root=str(out/'scores'),tests=h['tests'])
        for key in ('source_output','source_contract','restored_caches','frozen_sources'):scorer.pop(key)
        add('score',row,scorer,out/'scores/complete.json','SCORED_COMPLETE',[str(out/'predictions/complete.json')])
        refs[rid]={m:reference_paths['F0-'+m+'-s'+str(s)] for m in ('supervised','ordinary_PL','REPL')}
    write(root/'jobs.json',jobs);write(root/'reference_scores.json',refs);write(root/'sources.json',sources)
    return jobs
