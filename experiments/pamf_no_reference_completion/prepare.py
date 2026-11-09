"""Reuse verified frozen decisions and construct only missing stage jobs."""
from copy import deepcopy
from pathlib import Path
from collections import Counter

from experiments.cric_dura_comparison.io import read, write
from .plan import RUN, TASKS, VIEWS, rows, configuration


def historical(context, row, project):
    seed=row['model_seed']; root=Path(project)/'runs'
    if row['reuse_key'].startswith('ssl/'):
        folder=root/'20261027-phase1-pamf-paper-completion-m119-r03'
        key=f'main_split-structure-ssl-no_reference-permuted-s{seed}'
        old=read(folder/'sources.json')[key]
    elif row['reuse_key'].startswith('single/'):
        folder=root/'20261027-phase1-pamf-single-view-paper-tail-r01'/'pipeline'
        key=f'main_split-branch-no_reference-supervised-channel_aug-s{seed}'
        old=read(folder/'sources.json')[key]
    else: raise ValueError('Unknown historical reuse')
    source=Path(old['source_output']); pred=Path(old['predictions']); scores=Path(old['scores'])
    d=read(source/'complete.json'); c=d['config']; r=c['paper_row']
    if (d['status']!='SOURCE_FROZEN' or d['epoch']!=50 or d['updates']!=11100 or d['target_access'] or
            d['U_truth_read'] or d['head_fit_updates'] or c['model_seed']!=seed or
            read(c['source_contract'])!=read(context['handles']['F0']['source_contract']) or
            not (source/'final_ssdg.pth').is_file()):
        raise ValueError('CHECKPOINT_DATA_CONTRACT_MISMATCH: reuse')
    if row['reuse_key'].startswith('ssl/'):
        from experiments.pamf_paper_completion.plan import SSL_KEYS
        if r['structure']!='no_reference' or not r['permuted'] or r['signature_views']!=3 or r['learning_regime']!='ssl':
            raise ValueError('Wrong complete SSL graph')
        reference=context['templates'][str(seed)]['repl']
        if any(c.get(k)!=reference.get(k) for k in SSL_KEYS):raise ValueError('Reused full SSL recipe differs')
        actual=d['actual_U']['full_SSL'];segments=[(s['epoch'],s['updates'],s['config'].get('step_offset')) for s in d['segments']]
        if (actual['calls']!=6660 or actual['student_views']!=6*actual['seen'] or
                d['actual_U']['U_truth_read'] or d['actual_U']['EMA_updates'] or
                segments!=[(20,4440,0),(10,2220,4440),(20,4440,6660)]):
            raise ValueError('Reused full SSL execution differs')
    elif (r['arm']!='no_reference' or r['signature_views']!=1 or r['learning_regime']!='supervised' or
          c['ablation']!='F15' or c['U_use'] or c['satellite_CE_weight']!=.68):
        raise ValueError('Wrong single-view CE graph')
    pd=read(pred/'complete.json'); sd=read(scores/'complete.json')
    if (pd['status']!='ALL_PREDICTIONS_COMPLETE' or pd['truth_read'] or sd['status']!='SCORED_COMPLETE' or
            not set(TASKS).issubset(pd['tests']) or not set(row['views']).issubset(pd['views']) or
            not set(row['readouts']).issubset(pd['readouts'])):
        raise ValueError('Existing prediction or score incomplete')
    return dict(row=row,source_output=str(source),predictions=str(pred),scores=str(scores),reused=True)


def external_sources(rt, context):
    project=Path(rt['project']); old=project/'runs/20261023-phase1-practical-residual-control-r06'
    state=read(old/'pipeline_state.json'); jobs=read(old/'jobs.json')
    if state['status']!='ARTIFACTS_COMPLETE':raise ValueError('External panel not complete')
    matrix=read(Path(rt['release_root'])/'external_experiment.json')['matrix']['rows']
    expected={r['row_id']:r for r in matrix if r['kind']=='external'}
    if len(expected)!=528:raise ValueError('External panel size differs')
    matched={rid:{} for rid in expected}
    for j in jobs:
        if j['kind'] not in ('F','predict','score') or state['jobs'].get(j['id'],{}).get('status')!='COMPLETE':continue
        rid=j['id'].split('__',1)[1].split('--',1)[0]
        if rid not in expected:continue
        if j['kind'] in matched[rid]:raise ValueError('Duplicate complete external row')
        config=read(j['config']); d=read(j['completion']);r=expected[rid]
        if (any(config.get(k)!=r[k] for k in ('row_id','protocol_id','family','model_seed','cell')) or
                d['status'] not in j['successful_statuses']):raise ValueError('External scientific identity differs')
        if j['kind']=='F' and (d.get('target_access') or d.get('target_score_used')):
            raise ValueError('External source accessed target')
        if j['kind']=='predict' and (d['truth_read'] or d['query_fit'] or not set(TASKS).issubset(d['tests'])):
            raise ValueError('External frozen decisions incomplete')
        matched[rid][j['kind']]=dict(completion=j['completion'],output=str(Path(j['completion']).parent),config=j['config'])
    if any(set(v)!= {'F','predict','score'} for v in matched.values()):raise ValueError('External 528 complete rows missing')
    return [dict(row=expected[rid],source_output=v['F']['output'],predictions=v['predict']['output'],
                 scores=v['score']['output'],reused=True) for rid,v in matched.items()]


def prepare(rt):
    context=read(rt['context']);root=Path(rt['run_root']);jobs=[];sources={}
    (root/'configs').mkdir()
    def add(kind,row,c,complete,status,deps=()):
        p=root/'configs'/(kind+'__'+row['row_id']+'.json');write(p,c)
        jobs.append(dict(id=kind+'__'+row['row_id'],kind=kind,family=row['family'],learning=row['learning_regime'],
                         config=str(p),completion=str(complete),successful_statuses=[status],dependencies=list(deps)))
    for row in rows():
        if not row['new_training']:
            sources[row['row_id']]=historical(context,row,rt['project']);continue
        c=configuration(context,row,rt['project'],rt['code_commit']);c['frozen_sources']=rt['frozen_sources']
        src=Path(c['output_root']);add('train',row,c,src/'complete.json','SOURCE_FROZEN')
        out=root/'readouts'/row['row_id'];h=context['handles'][row['protocol_id']]
        if row['family']=='architecture':
            p=dict(run_id=RUN,row_id=row['row_id'],paper_row=row,variant=c['variant'],model_seed=row['model_seed'],
                   source_output=str(src),source_contract=c['source_contract'],capsule=context['ce_launch']['capsule'],
                   output_root=str(out/'predictions'),frozen_sources=rt['frozen_sources'])
            s={k:v for k,v in p.items() if k not in ('source_output','frozen_sources')}
            s.update(prediction_output=p['output_root'],truth=context['ce_launch']['truth'],output_root=str(out/'scores'),
                     double_cross_views_root=h['tests']['cross_both']['views_root'],classes=h['classes'])
            pred_status='PREDICTIONS_COMPLETE'
        else:
            p=dict(run_id=RUN,row_id=row['row_id'],paper_row=row,protocol_id=row['protocol_id'],model_seed=row['model_seed'],
                   source_output=str(src),source_contract=h['source_contract'],classes=h['classes'],source_days=[1,2,3],
                   tests={t:dict(views_root=h['tests'][t]['views_root']) for t in TASKS},
                   restored_caches={k:v['path'] for k,v in context['caches'][row['protocol_id']].items()} if row['family']=='factorial' else {},
                   views=row['views'],readouts=row['readouts'],output_root=str(out/'predictions'),frozen_sources=rt['frozen_sources'])
            s=deepcopy(p);s.update(prediction_output=p['output_root'],output_root=str(out/'scores'),tests={t:h['tests'][t] for t in TASKS})
            for key in ('source_output','restored_caches','frozen_sources'):s.pop(key)
            pred_status='ALL_PREDICTIONS_COMPLETE'
        add('predict',row,p,out/'predictions/complete.json',pred_status,[str(src/'complete.json')])
        add('score',row,s,out/'scores/complete.json','SCORED_COMPLETE',[str(out/'predictions/complete.json')])
        sources[row['row_id']]=dict(row=row,source_output=str(src),predictions=p['output_root'],scores=s['output_root'],reused=False)
    if len(sources)!=190 or len(jobs)!=552 or Counter(j['kind'] for j in jobs)!=dict(train=184,predict=184,score=184):
        raise ValueError('Registered 190 model matrix differs')
    write(root/'jobs.json',jobs);write(root/'sources.json',sources)
    external=external_sources(rt,context);write(root/'external_sources.json',external)
    return jobs
