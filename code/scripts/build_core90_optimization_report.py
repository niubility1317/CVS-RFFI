"""Rebuild report tables from frozen evidence; never access training or target IQ."""
from pathlib import Path
import ast
import csv
import json
import re

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / 'docs'
E = DOCS / 'evidence'
REPORT = DOCS / 'CORE90_OPTIMIZATION_REPORT_392005_20260911.md'

def read(path):
    return json.loads(path.read_text(encoding='utf-8'))

def table(headers, rows):
    return '\n'.join(['|'+'|'.join(headers)+'|', '|'+'|'.join(['---']*len(headers))+'|'] +
                     ['|'+'|'.join(map(str,row))+'|' for row in rows])

def main():
    scores = read(E/'core90_target25_392005_20260911.json')['scores']
    deep = read(E/'core90_c2_s4_full_analysis.json')
    scenes = ['clean','leo_clear_weak','leo_low_elev_weak','leo_rain_weak']
    assert len(scores)==25
    checks=0
    summary=[]
    for row,s in scores.items():
        assert s['complete'] and s['prediction_count']==672000
        assert set(s['scenes'])==set(scenes)
        for scene in scenes:
            v=s['scenes'][scene]
            assert v['count']==168000
            for grouping, n in [('per_rx',7),('per_day',4)]:
                groups=v[grouping]
                assert len(groups)==n and sum(x['count'] for x in groups.values())==v['count']
                value=sum(x['accuracy']*x['count'] for x in groups.values())/v['count']
                assert abs(value-v['accuracy'])<1e-12
                for g in groups.values():
                    assert sum(g['per_tx_count'].values())==g['count']
                    assert abs(sum(g['per_tx_accuracy'][k]*cnt for k,cnt in g['per_tx_count'].items())/g['count']-g['accuracy'])<1e-12
            assert set(v['per_tx_count'])==set(map(str,range(6)))
            assert sum(v['per_tx_count'].values())==v['count']
            assert abs(sum(v['per_tx_accuracy'][k]*cnt for k,cnt in v['per_tx_count'].items())/v['count']-v['accuracy'])<1e-12
            checks+=1
        leo=sum(s['scenes'][k]['accuracy'] for k in scenes[1:])/3
        assert abs(leo-s['leo_mean_accuracy'])<1e-12
        worst=min(x['accuracy'] for k in scenes[1:] for x in s['scenes'][k]['per_rx'].values())
        summary.append(dict(row=row,**{k:100*s['scenes'][k]['accuracy'] for k in scenes},
                            leo_mean=100*leo,delta_leo_pp=100*(leo-scores['B0']['leo_mean_accuracy']),
                            worst_leo_rx_scene=100*worst,clean_macro_f1=100*s['scenes']['clean']['macro_f1']))
    fields=list(summary[0])
    with (E/'core90_optimization_summary_392005.csv').open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(summary)
    text=REPORT.read_text(encoding='utf-8')
    def fill(key,value):
        nonlocal text
        start=f'<!-- {key} -->'; end=f'<!-- END {key} -->'
        text=re.sub(re.escape(start)+r'(?:[\s\S]*?'+re.escape(end)+r')?',
                    lambda _:start+'\n\n'+value+'\n\n'+end,text,count=1)
    tree=ast.parse((ROOT/'code/scripts/build_core90_game_matrix.py').read_text(encoding='utf-8'))
    fn=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='row_menu')
    assignment=next(x for x in fn.body if isinstance(x,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='rows' for t in x.targets))
    menu=ast.literal_eval(assignment.value)
    fill('MATRIX',table(['行','配置覆盖（其余同普通基础）','目标数据'],
                       [[row,'；'.join(f'`{k}={v}`' for k,v in spec[1].items()) or '仅源域审计，普通更新',
                         '已有四场景' if row in scores else '本报告未纳入'] for row,spec in menu.items()]))
    source_report=(DOCS/'CORE90_COMPLETED_RESULTS_392005_20260911.md').read_text(encoding='utf-8')
    source_table=source_report[source_report.index('|实验|Clean%|'):source_report.index('训练墙钟时间包含')]
    if '<!-- SOURCE -->' not in text:
        text=text.replace('## 6.全部25行目标结果','### 已完成15行的source V对照\n\n这里是同source接收机V的压力评估，不是目标接收机结果。其余行的source指标未在该15行汇总中统一列出，不以缺失值代替零。\n\n<!-- SOURCE -->\n\n## 6.全部25行目标结果')
    fill('SOURCE',source_table.strip())
    fill('RESULTS',table(['行','Clean%','晴空%','低仰角%','雨衰%','LEO均值%','ΔLEO(pp)','最弱LEO%','Clean宏F1%'],
                        [[x['row']]+[f'{x[k]:.2f}' for k in fields[1:]] for x in summary]))
    fill('ACTIVATION',table(['行','主更新','场求值','额外头步','审计数','训练h','峰值allocated GiB'],
          [[r,d['steps'],sum(int(k)*v for k,v in d['field_counts'].items()),d['head_steps'],d['audit_count'],
            f"{d['resources']['total_seconds']/3600:.3f}",f"{d['peak_training_cuda_bytes']/1024**3:.3f}"] for r,d in deep.items()]))
    original=(DOCS/'CORE90_C2_S4_FULL_ANALYSIS_20260911.md').read_text(encoding='utf-8')
    # Include the substantive technical sections; preserve their evidence links.
    excerpt=original[original.index('## 3.'):original.index('## 11.')]
    excerpt=re.sub(r'^(#{2,4}) ',lambda m:m.group(1)+'# ',excerpt,flags=re.M)
    fill('DEEP',excerpt)
    groups=[]
    for scene in scenes:
        groups.append('### '+scene)
        for key,title,prefix in [('per_rx','接收机','RX'),('per_day','日期','day'),('per_tx_accuracy','注册TX','TX')]:
            ids=sorted(scores['B0']['scenes'][scene][key],key=int)
            rows=[]
            for row,s in scores.items():
                values=s['scenes'][scene][key]
                rows.append([row]+[f"{100*(values[k] if key=='per_tx_accuracy' else values[k]['accuracy']):.2f}" for k in ids])
            groups.extend(['#### '+title,table(['行']+[prefix+k for k in ids],rows)])
    fill('GROUPS','\n\n'.join(groups))
    REPORT.write_text(text,encoding='utf-8',newline='\n')
    reread=REPORT.read_bytes().decode('utf-8')
    assert '\ufffd' not in reread and not reread.startswith('\ufeff')
    links=[]
    for link in re.findall(r'\]\(([^)]+)\)',reread):
        if '://' in link or link.startswith('#'): continue
        target=(REPORT.parent/link.split('#')[0]).resolve()
        assert target.exists(),str(target)
        links.append(link)
    verification={'status':'VERIFIED','scope':'frozen evidence aggregation; no new training or inference',
                  'target_rows':len(scores),'row_scenes_checked':checks,'prediction_count':sum(s['prediction_count'] for s in scores.values()),
                  'deep_log_rows':list(deep),'deep_epochs':sum(d['epochs'] for d in deep.values()),
                  'links_checked':len(links),'utf8':True,'report_bytes':len(REPORT.read_bytes()),
                  'excluded_target_row':'B6','unlaunched_dependency':'C4',
                  'live_remote_state_checked':False}
    (E/'core90_optimization_report_verification.json').write_text(json.dumps(verification,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(verification,ensure_ascii=False))

if __name__=='__main__':
    main()
