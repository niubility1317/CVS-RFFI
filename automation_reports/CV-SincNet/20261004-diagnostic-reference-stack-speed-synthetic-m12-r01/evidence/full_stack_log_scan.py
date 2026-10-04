from pathlib import Path
import sys,json
wt=Path('E:/type10-7/code/snapshots/daot_practical_three_20260918_wt');sys.path.insert(0,str(wt))
from experiments.cvs_phase1_stack.publish import ssh
from experiments.cvs_phase1_stack.design import BASE,PROJECT,PARENT_RUN,RUN
script=r'''
from pathlib import Path
import json,gzip,math,collections,time,csv,io,re,subprocess,statistics
root=Path(BASE);report=dict(started=time.time(),rows=[],r1_rows=[],gpu_samples=[])
def safe(v):
    if isinstance(v,float) and not math.isfinite(v):return None
    if isinstance(v,dict):return {str(k):safe(x) for k,x in v.items()}
    if isinstance(v,(list,tuple)):return [safe(x) for x in v]
    return v
def load(p):return json.loads(p.read_text()) if p.exists() else None
for source in sorted(root.glob('r*/source')):
    rid=source.parent.name;item=dict(row_id=rid,files={},structured={},stdout={},steps={})
    for p in source.iterdir():
        if p.is_file():item['files'][p.name]=dict(bytes=p.stat().st_size,mtime=p.stat().st_mtime)
    for name in ['resolved_config.json','resolved_native_args.json','initialization.json','completion.json']:
        item[name]=load(source/name)
    for name in ['metrics_epoch.jsonl','epoch_metrics.jsonl']:
        p=source/name;data=[]
        for line in p.read_text().splitlines():
            try:data.append(json.loads(line))
            except json.JSONDecodeError:pass
        item['structured'][name]=dict(count=len(data),first_epoch=data[0]['epoch'] if data else None,last_epoch=data[-1]['epoch'] if data else None,rows=data)
    for name in ['metrics_epoch.csv','epoch_metrics.csv']:
        p=source/name
        records=list(csv.DictReader(io.StringIO(p.read_text())))
        item['structured'][name]=dict(count=len(records),columns=list(records[0]) if records else [])
    p=Path(PROJECT)/'logs'/RUN/('source-'+rid+'.log');text=p.read_text(errors='replace');lines=text.splitlines()
    patterns={'traceback':r'Traceback \(most recent','oom':r'out of memory|CUDA error|Killed','nonfinite_fault':r'NONFINITE|FloatingPointError|nan loss|nonfinite loss','warnings':r'warning|Warning','stop_resume':r'early.stop|resume|restarting'}
    item['stdout']=dict(bytes=p.stat().st_size,lines=len(lines),matches={k:[dict(line=i+1,text=x[:1000]) for i,x in enumerate(lines) if re.search(pattern,x)] for k,pattern in patterns.items()})
    p=source/'step_metrics.jsonl.gz';stats={};count=0;tail='complete';first_step=None;last_step=None
    try:
        with gzip.open(p,'rt',encoding='utf-8') as f:
            for line in f:
                if not line.endswith('\n'):tail='incomplete trailing record';break
                r=json.loads(line);count+=1;first_step=first_step or r['step'];last_step=r['step'];ep=r['epoch'];g=stats.setdefault(ep,dict(count=0,numeric={}))
                g['count']+=1
                for k,v in r['metrics'].items():
                    if isinstance(v,(int,float)) and math.isfinite(v):
                        a=g['numeric'].setdefault(k,[0.,v,v,0]);a[0]+=v;a[1]=min(a[1],v);a[2]=max(a[2],v);a[3]+=1
    except EOFError:tail='live gzip member not closed; all available complete lines parsed'
    item['steps']=dict(count=count,first_step=first_step,last_step=last_step,tail=tail,epochs={str(ep):dict(count=g['count'],metrics={k:dict(mean=v[0]/v[3],min=v[1],max=v[2],count=v[3]) for k,v in g['numeric'].items()}) for ep,g in stats.items()})
    report['rows'].append(item)
for source in sorted((Path(PROJECT)/'runs'/PARENT_RUN).glob('*/source')):
    p=source/'epoch_metrics.jsonl'
    if not p.exists():continue
    report['r1_rows'].append(dict(row_id=source.parent.name,epochs=[json.loads(v) for v in p.read_text().splitlines() if v.strip()],resolved=load(source/'resolved_config.json'),completion=load(source/'completion.json')))
report['parent_queue']=load(root/'queue_state.json');report['parent_failure']=load(root/'failure.json')
report['gpu_snapshot']=subprocess.check_output(['nvidia-smi','--query-gpu=index,utilization.gpu,memory.used,memory.free,power.draw','--format=csv,noheader,nounits'],text=True)
report['finished']=time.time();print(json.dumps(safe(report),ensure_ascii=True,allow_nan=False))
'''.replace('BASE',repr(BASE)).replace('PROJECT',repr(PROJECT)).replace('PARENT_RUN',repr(PARENT_RUN)).replace('RUN',repr(RUN))
data=json.loads(ssh(script));out=wt/'local_artifacts/cvs_reference_stack_speed_20261004_r01';out.mkdir(parents=True,exist_ok=True)
(out/'full_log_scan.json').write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(dict(seconds=data['finished']-data['started'],rows=[dict(row_id=r['row_id'],epochs=r['structured']['epoch_metrics.jsonl']['count'],steps=r['steps']['count'],tail=r['steps']['tail'],stdout_lines=r['stdout']['lines'],errors={k:len(v) for k,v in r['stdout']['matches'].items()}) for r in data['rows']],r1_rows=len(data['r1_rows']),failure=data['parent_failure'])))
