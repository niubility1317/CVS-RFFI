"""Reuse only complete same-contract sources and their frozen decisions."""
from copy import deepcopy
from pathlib import Path
from experiments.cric_dura_comparison.io import read,write
from .plan import RUN,TASKS,READOUTS,configuration,make_plan,SSL_KEYS

def validate_reused_SSL(done,reference):
    c=done['config']
    for k in SSL_KEYS:
        if c.get(k)!=reference.get(k):raise ValueError('Reused full U recipe differs: '+k)
    actual=done.get('actual_REPL',{}).get('calls',done.get('actual_adapter_SSL',{}).get('hard_PL_calls'))
    segments=[(s['epoch'],s['updates'],s['config'].get('step_offset')) for s in done['segments']]
    if actual!=6660 or segments!=[(20,4440,0),(10,2220,4440),(20,4440,6660)]:raise ValueError('Reused full SSL execution schedule differs')

def reuse(ctx,row,rt):
    key=row['reuse_key']
    if key.startswith('current/'):
        old=ctx['current_views'][key.removeprefix('current/')];full=False
    else:
        old=ctx['existing'][key];full=True
    folder=Path(old['source_output']);d=read(folder/'complete.json');c=d['config'];h=ctx['handles'][row['protocol_id']]
    if (d['status']!='SOURCE_FROZEN' or d['updates']!=11100 or d['epoch']!=50 or d['target_access'] or d['U_truth_read'] or d['head_fit_updates']
        or c['model_seed']!=row['model_seed'] or read(c['source_contract'])!=read(h['source_contract'])):raise ValueError('CHECKPOINT_DATA_CONTRACT_MISMATCH: reuse')
    if not full:
        if c['view_comparison']['view_count']!=row['signature_views'] or not row['permuted']:raise ValueError('Matching permuted view row required')
        predictions=Path(old['predictions']);scores=None
    else:
        expected={'sup':'F20','pl':'F21'}.get(row['method'],'F21')
        if c['ablation']!=expected:raise ValueError('Wrong reused objective')
        arm=c.get('repl_row',{}).get('structure','full')
        if arm!={'full':'full','no_signature':'NO_SIGNATURE','moment':'COV_X3'}[row['structure']]:raise ValueError('Wrong reused architecture')
        if row['learning_regime']=='ssl':
            reference=ctx['templates'][str(row['model_seed'])]['repl']
            validate_reused_SSL(d,reference)
        predictions=Path(old['predictions']);scores=Path(old['scores'])
        sd=read(scores/'complete.json');pd=read(predictions/'complete.json')
        if sd['status']!='SCORED_COMPLETE' or pd['status']!='ALL_PREDICTIONS_COMPLETE' or pd['truth_read']:raise ValueError('Historical readouts incomplete')
    return folder,predictions,scores

def prepare(rt):
    ctx=read(rt['context']);root=Path(rt['run_root']);jobs=[];sources={}
    def add(kind,row,c,completion,status,deps=()):
        path=root/'configs'/(kind+'__'+row['row_id']+'.json');write(path,c)
        jobs.append(dict(id=kind+'__'+row['row_id'],kind=kind,family=row['family'],learning=row['learning_regime'],config=str(path),completion=str(completion),
                         successful_statuses=[status],dependencies=list(deps)))
    declared=read(Path(rt['release_root'])/'experiment.json')
    rows=[r['paper_row'] for r in declared['rows']]
    if rows!=make_plan()['rows']:raise ValueError('Registered execution matrix differs from code')
    for row in rows:
        c=configuration(ctx,row,rt['project'],rt['code_commit']);c['frozen_sources']=rt['frozen_sources']
        reused_pred=None
        if row['reuse_key']:
            folder,reused_pred,old_scores=reuse(ctx,row,rt)
            if old_scores is not None:
                sources[row['row_id']]=dict(row=row,source_output=str(folder),predictions=str(reused_pred),scores=str(old_scores),reused=True,readouts_reused=True)
                continue
        else:
            folder=Path(c['output_root']);add('train',row,c,folder/'complete.json','SOURCE_FROZEN')
        out=root/'readouts'/row['row_id'];h=ctx['handles'][row['protocol_id']]
        if row['family']=='architecture':
            p=dict(run_id=RUN,row_id=row['row_id'],paper_row=row,variant=c['variant'],model_seed=row['model_seed'],
                   source_output=str(folder),source_contract=c['source_contract'],capsule=ctx['ce_launch']['capsule'],
                   output_root=str(out/'predictions'),frozen_sources=rt['frozen_sources'])
            s={k:v for k,v in p.items() if k not in ('source_output','frozen_sources')}
            s.update(prediction_output=p['output_root'],truth=ctx['ce_launch']['truth'],output_root=str(out/'scores'),
                     double_cross_views_root=h['tests']['cross_both']['views_root'],classes=h['classes'])
            pred_status='PREDICTIONS_COMPLETE'
        else:
            p=dict(run_id=RUN,row_id=row['row_id'],paper_row=row,protocol_id=row['protocol_id'],model_seed=row['model_seed'],
                source_output=str(folder),source_contract=h['source_contract'],classes=h['classes'],source_days=[1,2,3],
                tests={t:dict(views_root=h['tests'][t]['views_root']) for t in TASKS},
                restored_caches={k:v['path'] for k,v in ctx['caches'][row['protocol_id']].items()},
                views=row['views'],readouts=list(READOUTS),reuse_predictions=str(reused_pred) if reused_pred else None,
                output_root=str(out/'predictions'),frozen_sources=rt['frozen_sources'])
            s=deepcopy(p);s.update(prediction_output=p['output_root'],output_root=str(out/'scores'),tests={t:h['tests'][t] for t in TASKS})
            for k in ('source_output','restored_caches','frozen_sources','reuse_predictions'):s.pop(k,None)
            pred_status='ALL_PREDICTIONS_COMPLETE'
        add('predict',row,p,out/'predictions/complete.json',pred_status,[str(folder/'complete.json')])
        add('score',row,s,out/'scores/complete.json','SCORED_COMPLETE',[str(out/'predictions/complete.json')])
        sources[row['row_id']]=dict(row=row,source_output=str(folder),predictions=p['output_root'],scores=s['output_root'],
                                   reused=bool(row['reuse_key']),readouts_reused=False,raw_decisions_reused=bool(reused_pred))
    if len(jobs)!=369 or sum(j['kind']=='train' for j in jobs)!=119 or len(sources)!=200:raise ValueError('Deduplicated completion matrix differs')
    write(root/'jobs.json',jobs);write(root/'sources.json',sources)
    return jobs
