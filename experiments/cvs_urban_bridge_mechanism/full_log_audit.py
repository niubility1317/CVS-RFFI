"""Full new logs plus exact archived audits; never load IQ, weights or truth."""
import argparse,copy,importlib.util,json,subprocess
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))


def run(output,legacy):
    spec=read(ROOT/'experiments/cvs_urban_bridge50/registration.json');base=spec['execution']['remote_run_root']
    fresh=[dict(arm=r['config']['arm'],row_id=r['row_id'],source=r['config']['output_root'],log=r['log_path'],
        config=r['config'],HQ_profile=True) for r in spec['rows']]
    if [r['arm'] for r in fresh]!=['BRIDGE_CE','BRIDGE_SIG']:raise ValueError('Only registered two-row pilot')
    module_spec=importlib.util.spec_from_file_location('bridge_audit_operations',ROOT/'experiments/cvs_urban_sigtime/operations.py')
    operations=importlib.util.module_from_spec(module_spec);module_spec.loader.exec_module(operations)
    program=(ROOT/'experiments/cvs_urban_semantic50/_audit_worker.py').read_text(encoding='utf-8-sig')+'\n'
    program+=Path(__file__).with_name('_audit_worker.py').read_text(encoding='utf-8-sig')+'\n'
    program+='print(json.dumps(audit_bridge('+repr(dict(base=base,rows=fresh))+',audit_row)))\n'
    result=operations.remote(program)
    if result['status']!='COMPLETE_NEW_LOG_AUDIT':
        print(json.dumps(dict(status=result['status'],run_id=spec['run_id'],no_partial_curve_claim=True)));return result
    old=read(legacy)
    if old['status']!='COMPLETE_TRAINING_EVIDENCE' or len(old['rows'])!=5:raise ValueError('Original complete five-condition semantic audit required')
    records=list(result['rows']);refs=spec['checkpoint']['archived_comparison_configs']
    for name,original_arm in (('CLEAN','D0'),('NOMINAL','D1'),('POSITIVE_J','D2')):
        value=copy.deepcopy(next(r for r in old['rows'] if r['arm']==original_arm));c=refs[name]
        if value['completion']['config']!=c or value['complete_steps']!=11100:raise ValueError('Exact archived source/default-head audit differs')
        value.update(actual_arm=original_arm,arm=name,reused=True,new_log_steps_read=0,
            reused_audit_ref=str(legacy)+'#rows/'+original_arm,actual_source=c['output_root'])
        records.append(value)
    combined=dict(status='COMPLETE_TRAINING_EVIDENCE',run_id=spec['run_id'],rows=records,
        new_full_log_steps_read=22200,reused_verified_steps=33300,total_update_evidence=55500,
        completion=result['completion'],target_performance_used_for_training=False,original_logs_reanalyzed=False,
        scope='Two complete new five-view logs; three exact previously audited source controls reused unchanged')
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    (out/'training_audit.json').write_text(json.dumps(combined,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    for name in ('summary.json','resources.json','training_curves.json','analysis.md','complete.json','source_family_decision.json'):
        subprocess.run(['scp',*operations.SSH,'N607:'+base+'/'+name,str(out/name)],check=True,timeout=60)
    configs=dict(refs,**{r['config']['arm']:r['config'] for r in spec['rows']})
    for arm,c in configs.items():
        remote_root=c['output_root'].rsplit('/',1)[0];destination=out/'formal_rows'/arm
        (destination/'scoring').mkdir(parents=True,exist_ok=True)
        for name in ('report.md','scoring/scores.json','scoring/complete.json'):
            subprocess.run(['scp',*operations.SSH,'N607:'+remote_root+'/'+name,str(destination/name)],check=True,timeout=60)
    print(json.dumps(dict(status=combined['status'],new_steps=22200,reused_steps=33300,output=str(out))));return combined


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True);p.add_argument('--legacy-audit',required=True)
    a=p.parse_args();run(a.output,a.legacy_audit)
