"""Export aggregate-safe scored metrics and failure categories for the Phase2 report.

Reads only fixed prediction score metadata, never target IQ, predictions or truth.
"""
import argparse
from collections import Counter
from functools import lru_cache
import gzip
import json
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def export_author(base, runid):
    run=base/'runs'/runid
    config=read(base/'preparations'/runid/'scheduler.json')
    finish=read(run/'scheduler'/'complete.json')
    assert finish['truth_last'] is True
    quarantined=set(finish.get('quarantined',[]))
    rows=[];failures=[];scored=0
    for job in config['jobs']:
        if job['id'] in quarantined:
            marker=read(Path(job['output'])/'numerical_failure.json')
            assert marker['truth_read'] is False and marker['query_fit'] is False
            failures.append(dict(id=job['id'],profile=job['profile'],capsule=job['capsule'],
                seed=job['seed'],status=marker['status'],error=marker['error']))
            continue
        score=Path(job['score']);complete=read(score/'complete.json');data=read(score/'metrics.json')
        assert complete['status']=='SCORED_COMPLETE' and complete['truth_last'] is True
        assert data['truth_last'] is True and data['target_feedback'] is False
        assert len(data['rows'])==complete['rows'] and data['model_seed']==job['seed']
        _,pool,episode,receiver=job['capsule'].split('/')
        for row in data['rows']:
            assert row['stage']=='C' and row['k'] in (5,10) and row['new_count'] in (5,10)
            assert (row['physical_pool'],row['episode'],row['receiver'])==(pool,episode,receiver)
            res=row.get('resources') or {};meta=res.get('C_protocol') or res.get('native_resources') or {}
            assert meta.get('mopc_sampling_version')=='author_drop_last_v1'
            assert 'clip' not in row['arm'].lower()
            metrics=row['metrics']
            rows.append(dict(profile=job['profile'],capsule=job['capsule'],seed=job['seed'],
                k=row['k'],n=row['new_count'],split_id=row['split_id'],count=metrics['count'],
                registered_classes=row['registered_classes'],input_condition=row['input_condition'],
                accuracy=metrics['accuracy'],macro_accuracy=metrics['macro_accuracy'],
                old=metrics['old_macro_accuracy'],new=metrics['new_macro_accuracy'],
                h=metrics['H_macro'],f1=metrics['macro_f1'],floor=metrics['class_floor']))
        scored+=1
    assert scored==finish['scored'] and len(quarantined)==len(failures)
    assert scored+len(failures)==len(config['jobs'])
    return dict(run=runid,status=finish['status'],jobs=len(config['jobs']),scored=scored,
                failures=failures,rows=rows)


@lru_cache(maxsize=128)
def score_file(path):
    score=read(path)
    assert score['truth_last'] is True and score['target_feedback'] is False
    return score


def export_clean_pamf(path):
    rows=[];seen=set()
    with gzip.open(path,'rt',encoding='utf8') as stream:
        for line in stream:
            r=json.loads(line)
            if r.get('group')!='pamf' or r.get('method') not in ('PAMF_clean','PAMF_恢复') or \
               r.get('episode') not in ('MIX01','MIX02','MIX03') or r.get('seed') not in (2026092701,2026092702,2026092703) or \
               r.get('k') not in (5,10) or (r.get('stage')=='C' and r.get('n') not in (5,10)) or \
               r.get('stage') not in ('A0','B','C'):
                continue
            stage=r['stage'];pathcode=('Q0' if r['method']=='PAMF_clean' else 'Q1') if stage=='A0' else \
                ('S0Q0' if r['method']=='PAMF_clean' else 'S1Q1')
            matched=[x for x in score_file(r['score_source'])['rows'] if x['stage']==stage and x['k']==r['k'] and
                     x['new_count']==r['n'] and x['input_path']==pathcode and x['arm']=='R42']
            assert len(matched)==1
            item=matched[0];metric=item['metrics'];cap=f"capsules/{r['pool']}/{r['episode']}/{r['receiver']}"
            assert item['registered_classes']==r['classes'] and metric['count']==r['count']
            key=(r['method'],cap,r['seed'],stage,r['k'],r['n']);assert key not in seen;seen.add(key)
            assert item['input_condition']=='clean_ground_pre_satellite_normalized'
            rows.append(dict(method=r['method'],capsule=cap,seed=r['seed'],stage=stage,k=r['k'],n=r['n'],
                split_id=item['split_id'],registered_classes=item['registered_classes'],count=metric['count'],
                accuracy=metric['accuracy'],macro_accuracy=metric['macro_accuracy'],old=metric['old_macro_accuracy'],
                new=metric['new_macro_accuracy'],h=metric['H_macro'],f1=metric['macro_f1'],floor=metric['class_floor']))
    return rows


def main():
    p=argparse.ArgumentParser();p.add_argument('--base',required=True);p.add_argument('--clean-snapshot',required=True)
    p.add_argument('--output',required=True);a=p.parse_args();base=Path(a.base)
    ids=['20261118-phase2-mopc-author-sat-pilot-s2026092701-r03',
         '20261118-phase2-mopc-author-sat-full-m3-r02',
         '20261118-phase2-mopc-author-clean-pilot-m3-r01',
         '20261118-phase2-mopc-author-clean-full-mix3-m3-r02']
    runs=[export_author(base,name) for name in ids];pamf=export_clean_pamf(a.clean_snapshot)
    out=dict(runs=runs,pamf_clean=pamf,metric_source='independent truth-last score metadata; no predictions or truth')
    Path(a.output).write_text(json.dumps(out,ensure_ascii=False),encoding='utf8')
    print(json.dumps(dict(runs=[dict(run=x['run'],scored=x['scored'],quarantined=len(x['failures']),
        rows=len(x['rows']),failure_types=dict(Counter(f['status'] for f in x['failures']))) for x in runs],
        pamf_clean_rows=len(pamf)),ensure_ascii=False))


if __name__=='__main__':main()
