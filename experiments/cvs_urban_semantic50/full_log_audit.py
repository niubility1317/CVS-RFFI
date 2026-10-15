"""Read all new completed logs; reuse the unchanged original A0/A-J full audits."""
import argparse,copy,importlib.util,json,subprocess
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))


def run(output,legacy):
    spec=read(ROOT/'experiments/cvs_urban_semantic50/registration.json')
    rows=spec['rows'];base=spec['execution']['remote_run_root']
    fresh=[dict(arm=r['config']['arm'],row_id=r['row_id'],source=r['config']['output_root'],log=r['log_path'],config=r['config'])
        for r in rows if not r['config']['reused']]
    if [r['arm'] for r in fresh]!=['D0','D3','D4']:raise ValueError('Only the three registered corrected rows may be audited')
    module_spec=importlib.util.spec_from_file_location('semantic_audit_operations',ROOT/'experiments/cvs_urban_sigtime/operations.py')
    ops=importlib.util.module_from_spec(module_spec);module_spec.loader.exec_module(ops)
    program=Path(__file__).with_name('_audit_worker.py').read_text(encoding='utf-8-sig')
    program+='\nprint(json.dumps(audit('+repr(dict(base=base,rows=fresh))+')))\n'
    result=ops.remote(program)
    if result['status']!='COMPLETE_NEW_LOG_AUDIT':
        print(json.dumps(dict(status=result['status'],run_id=spec['run_id'],no_partial_curve_claim=True)))
        return result
    old=read(legacy)
    if old['status']!='COMPLETE_LOG_AUDIT' or len(old['rows'])!=8:raise ValueError('Need the existing complete eight-position audit')
    records=list(result['rows']);reused=[]
    for row in rows:
        c=row['config']
        if not c['reused']:continue
        original=c['reuse_source_config'];value=next(r for r in old['rows'] if r['row_id']==original['row_id'])
        if value['completion']['config']!=original or value['complete_steps']!=11100:
            raise ValueError('Original audited control differs from exact registered provenance')
        value=copy.deepcopy(value);actual_id=value['row_id']
        value.update(arm=c['arm'],row_id=row['row_id'],actual_row_id=actual_id,actual_source=c['output_root'],reused=True,
            reused_audit_ref=str(legacy)+'#rows/'+actual_id,new_log_steps_read=0)
        records.append(value);reused.append(row['row_id'])
    if len(records)!=5 or {r['arm'] for r in records}!={'D0','D1','D2','D3','D4'}:
        raise ValueError('Five-row matched evidence incomplete')
    records.sort(key=lambda r:r['arm']);out=Path(output);out.mkdir(parents=True,exist_ok=True)
    combined=dict(status='COMPLETE_TRAINING_EVIDENCE',run_id=spec['run_id'],rows=records,
        new_full_log_steps_read=result['new_updates_parsed'],reused_verified_steps=sum(r['complete_steps'] for r in records if r['reused']),
        total_update_evidence=sum(r['complete_steps'] for r in records),completion=result['completion'],
        target_performance_used_for_training=False,original_logs_reanalyzed=False,
        scope='Three new complete logs plus two unchanged original audited controls; native stdout/numeric step/epoch/update evidence retained')
    (out/'training_audit.json').write_text(json.dumps(combined,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    for name in ('summary.json','resources.json','training_curves.json','analysis.md','complete.json','source_family_decision.json'):
        subprocess.run(['scp',*ops.SSH,'N607:'+base+'/'+name,str(out/name)],check=True,timeout=60)
    print(json.dumps(dict(status=combined['status'],new_steps=combined['new_full_log_steps_read'],reused_steps=combined['reused_verified_steps'],
        rows=[r['row_id'] for r in records],output=str(out))))
    return combined


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True,type=Path)
    p.add_argument('--legacy-audit',required=True,type=Path);a=p.parse_args();run(a.output,a.legacy_audit)
