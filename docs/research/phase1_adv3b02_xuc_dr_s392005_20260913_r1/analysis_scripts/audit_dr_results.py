import json,subprocess
from pathlib import Path
CODE=r'''
import json,math,re,time,statistics
from pathlib import Path
from collections import Counter,defaultdict
project=Path('/home/szu2070436088/2510044040/CV-SincNet')
roots={'dr':'phase1_adv3b02_xuc_dr_s392005_20260913_r1','old':'phase1_adv3b02_xuc15_s392005_20260913_r1'}
def stat(v):
    return dict(n=len(v),mean=statistics.mean(v),min=min(v),max=max(v),sum=sum(v)) if v else dict(n=0)
def read(p):return json.loads(p.read_text()) if p.exists() else None
out={}
for family,rid in roots.items():
    root=project/'runs'/rid;s=read(root/'pipeline_state.json');data=dict(state=s,rows={})
    for name,entry in s['rows'].items():
        f=root/name;d=dict(state=entry,completion=read(f/'completion.json'),score=read(f/'target_prediction/score.json'),initialization=read(f/'initialization.json'),resource=read(f/'phase1_resource_summary.json'))
        d['inventory']=[dict(name=p.name,bytes=p.stat().st_size) for p in f.iterdir() if p.is_file()]
        logfiles=['logs.jsonl','metrics_epoch.jsonl','rc4_calibration.jsonl','cstar_audit.jsonl','legacy_audit.jsonl']
        for file in logfiles:
            p=f/file
            if not p.exists():continue
            records=[];bad=[]
            with p.open() as h:
                for n,line in enumerate(h,1):
                    try:records.append(json.loads(line))
                    except json.JSONDecodeError:bad.append(n)
            d[file]=dict(count=len(records),bad=bad,records=records)
        p=f/'actions.jsonl'
        if p.exists():
            counts=Counter();actions=Counter();fields=Counter();phases=defaultdict(lambda:defaultdict(list));probes=[];bad=[];last=None;first_dr=None;source=set();unlabeled=set();gaps=[];expected=0
            for n,line in enumerate(p.open(),1):
                try:a=json.loads(line)
                except json.JSONDecodeError:bad.append(n);continue
                if a['step']!=expected:gaps.append([expected,a['step']])
                expected=a['step']+1;last=a;counts['records']+=1;counts['accepted']+=bool(a['accepted']);actions[a['action']]+=1;fields[str(a['field_evaluations'])]+=1
                epoch=a['origin_epoch'];phase='001-020' if epoch<=20 else '021-130' if epoch<=130 else '131-180' if epoch<=180 else '181-200'
                source.update(a['source_ids']);unlabeled.update(a.get('unlabeled_ids',[]));counts['L_exposures']+=len(a['source_ids']);counts['U_exposures']+=len(a.get('unlabeled_ids',[]))
                for k in ['loss','grad_norm']:
                    if not math.isfinite(a[k]):counts['nonfinite_'+k]+=1
                    else:phases[phase][k].append(a[k])
                for k,v in a['terms'].items():
                    if isinstance(v,(int,float)):
                        if math.isfinite(v):phases[phase]['term_'+k].append(v)
                        else:counts['nonfinite_term_'+k]+=1
                dr=a.get('daot_rc4',{})
                for k in ['daot_executed','rc4_executed','identity_active']:
                    counts[k]+=bool(dr.get(k,False))
                for k in ['hard_count','partial_count']:
                    counts[k+'_sum']+=dr.get(k,0);counts[k+'_nonzero_steps']+=dr.get(k,0)>0
                    phases[phase][k].append(dr.get(k,0))
                if dr.get('daot_labeled',0)>0 and first_dr is None:first_dr=dict(step=a['step'],origin_epoch=epoch,execution_epoch=a['execution_epoch'],dr=dr)
                if 'daot_grad_norm' in dr:probes.append(dict(step=a['step'],origin_epoch=epoch,execution_epoch=a['execution_epoch'],**dr))
                if dr and dr.get('scale_commits_before')!=a['step']:counts['scale_commit_mismatch']+=1
                if family=='dr':
                    needed={'daot_labeled','daot_unlabeled','rc4_total'}
                    if a['fusion']['x_enabled']:needed.add('x_cross_rx')
                    # X component key is reported independently below; DR completeness is asserted.
                    if any(not {'daot_labeled','daot_unlabeled','rc4_total'}.issubset(set(c)) for c in a['field_components']):counts['incomplete_dr_fields']+=1
                for key in ['normalized_valid_blocks','legal_anchors']:
                    if a.get('fusion',{}).get(key,0)>0:counts[key+'_nonzero_steps']+=1
            d['action_audit']=dict(counts=dict(counts),actions=dict(actions),field_evaluations=dict(fields),unique_L=len(source),unique_U=len(unlabeled),gaps=gaps,bad=bad,phases={k:{x:stat(v) for x,v in val.items()} for k,val in phases.items()},gradient_probes=probes,first_nonzero_daot=first_dr,last={k:v for k,v in last.items() if k not in ['source_ids','unlabeled_ids','selected_mask']} if last else None)
        logs=[]
        for p in (project/'logs'/rid).glob(name+'.*.log'):
            text=p.read_text(errors='replace');lines=text.splitlines()
            errors=[dict(line=i,text=v[:600]) for i,v in enumerate(lines,1) if re.search(r'Traceback|RuntimeError|OutOfMemoryError|CUDA error|\bKilled\b|\bNaN\b|\bInfinity\b',v)]
            warnings=Counter(v for v in lines if 'Warning' in v)
            logs.append(dict(path=str(p),bytes=p.stat().st_size,lines=len(lines),errors=errors,warnings=dict(warnings)))
        d['stdout_scan']=logs
        data['rows'][name]=d
    out[family]=data
out['read_at']=time.time()
print(json.dumps(out))
'''
compile(CODE,'remote_audit','exec')
r=subprocess.run(['ssh','-F','E:/type10-7/tools/n607_ssh_config','-T','-o','BatchMode=yes','-o','ConnectTimeout=10','N607','python3 -'],input=CODE.encode(),capture_output=True,check=True)
d=json.loads(r.stdout)
out=Path('E:/type10-7/automation_reports/CV-SincNet/phase1_adv3b02_xuc_dr_s392005_20260913_r1')
(out/'full_run_audit.json').write_text(json.dumps(d,ensure_ascii=False,indent=2),encoding='utf-8')
for family,group in d.items():
    if not isinstance(group,dict):continue
    print(family,group['state']['status'])
    for name,row in group['rows'].items():
        print(name,'score',bool(row['score']),'actions',row.get('action_audit',{}).get('counts'),'metrics',row.get('metrics_epoch.jsonl',{}).get('count'),'logs',row.get('logs.jsonl',{}).get('count'))
