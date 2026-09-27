"""Read complete relevant logs; no remote mutation, model load, or target access."""
from pathlib import Path
import subprocess,json,gzip,collections,statistics,sys
R=Path('E:/type10-7');A=Path(__file__).resolve().parent
P=R/'automation_reports/CV-SincNet/20260920-phase1-daot-rc4-residual-ratios-manysig-s392005-r01'
historical=json.loads((P/'evidence/final_full_snapshot_20260921.json').read_text(encoding='utf-8'))
summary=[]
for x in historical['rows']:
    rows=x['files'].get('metrics_epoch.jsonl',{}).get('records',[])
    if not rows:continue
    out={'row':x.get('receipt',{}).get('row_id'),'epochs':len(rows),'keys':list(rows[-1]),'last':rows[-1]}
    out['sequence_continuous']=[r['epoch'] for r in rows]==list(range(1,len(rows)+1))
    out['phase_means']={}
    fields=[k for k in rows[-1] if any(s in k for s in ['seconds','elapsed','daot','rc4','sat_cls','pseudo','coverage','skipped','validation','val_acc'])]
    for lo,hi in [(1,20),(21,40),(41,79),(80,90),(91,129),(130,200)]:
        rr=[r for r in rows if lo<=r['epoch']<=hi]
        if rr:out['phase_means'][f'{lo}-{hi}']={k:statistics.mean(float(r[k]) for r in rr if isinstance(r.get(k),(float,int))) for k in fields if any(isinstance(r.get(k),(float,int)) for r in rr)}
    summary.append(out)
(A/'historical_log_analysis.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
print('Historical complete epoch records',sum(r['epochs'] for r in summary),[(r['row'],r['epochs']) for r in summary])
payload=r'''
from pathlib import Path
import json,datetime,re,subprocess
root=Path('/home/szu2070436088/2510044040/CV-SincNet')
run='20260927-phase1-cvs-daot-rc4-practical-manysig-m5-r01'
result={'read_at':datetime.datetime.now().astimezone().isoformat(),'run':run,'rows':[],'processes':[]}
for p in sorted((root/'runs'/run).glob('cvs-daot-rc4-*')):
    x={'row':p.name,'files':{}}
    for name in ['metrics_epoch.jsonl','resolved_config.json','startup.json','initialization.json','completion.json']:
        q=p/name
        if not q.exists():continue
        raw=q.read_text(encoding='utf-8');x['files'][name]=[json.loads(line) for line in raw.splitlines() if line.strip()] if name.endswith('.jsonl') else json.loads(raw)
    q=root/'logs'/run/(p.name+'.train.log')
    if q.exists():
        lines=q.read_text(encoding='utf-8',errors='replace').splitlines()
        matches=[{'line':i+1,'text':line[:700]} for i,line in enumerate(lines) if re.search(r'Traceback|Error|Warning|Killed|out of memory|\bnan\b|\binf\b',line,re.I)]
        x['stdout']={'path':str(q),'lines':len(lines),'matches':matches}
    q=p/'final_ssdg.pth';x['final_checkpoint_bytes']=q.stat().st_size if q.exists() else None
    result['rows'].append(x)
for p in Path('/proc').iterdir():
    if not p.name.isdigit():continue
    try:
        argv=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')
        if run in argv:result['processes'].append({'pid':int(p.name),'argv':argv,'cwd':str((p/'cwd').resolve())})
    except (OSError,PermissionError):pass
print(json.dumps(result,ensure_ascii=False,allow_nan=True))
'''
compile(payload,'remote_readonly','exec')
p=subprocess.run(['ssh','-F',str(R/'tools/n607_ssh_config'),'-T','-o','BatchMode=yes','-o','ConnectTimeout=10','N607','python3','-'],input=payload.encode('utf-8'),capture_output=True,timeout=90)
if p.returncode:raise RuntimeError(p.stderr.decode(errors='replace'))
remote=json.loads(p.stdout);(A/'current_source_logs.json.gz').write_bytes(gzip.compress(p.stdout))
compact=[]
for x in remote['rows']:
    rr=x['files'].get('metrics_epoch.jsonl',[])
    cfg=x['files'].get('resolved_config.json',{})
    compact.append({'row':x['row'],'epochs':len(rr),'last_epoch':rr[-1].get('epoch') if rr else None,'last':rr[-1] if rr else {},'final_checkpoint_bytes':x['final_checkpoint_bytes'],'stdout_lines':x.get('stdout',{}).get('lines'),'stdout_matches':len(x.get('stdout',{}).get('matches',[])),'execution_flags':{k:v for k,v in cfg.items() if any(s in k for s in ['practical','identity','gradient_snapshot','execution','source_validation_reuse','telemetry'])}})
(A/'current_source_summary.json').write_text(json.dumps({'read_at':remote['read_at'],'rows':compact,'processes':remote['processes']},ensure_ascii=False,indent=2),encoding='utf-8')
print('Current source logs',remote['read_at'],[(x['row'],x['epochs'],x['final_checkpoint_bytes'],x['stdout_lines']) for x in compact])
print('Flags',compact[0]['execution_flags'] if compact else {})
