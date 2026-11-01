"""Summarize completed score artifacts only; never score or read query/truth."""
import argparse
import csv
import datetime as dt
import html
import json
from pathlib import Path
import statistics

PATHS=('S0Q0','S0Q1','S0Q2','S1Q0','S1Q1','S1Q2')
UNIT=('physical_pool','episode','receiver','model_seed','k','new_count')


def load(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def aggregate(payload,cohort,expected_units):
    new={tuple(r[k] for k in UNIT)+(r['input_path'],):r for r in payload['new'] if r['stage']=='C'}
    old={tuple(r[k] for k in UNIT)+(r['input_path'],):r for r in payload['baseline'] if r['stage']=='C'}
    assert len(new)==sum(r['stage']=='C' for r in payload['new']), 'Duplicate R42 C row'
    assert len(old)==sum(r['stage']=='C' for r in payload['baseline']), 'Duplicate baseline C row'
    groups={}
    for key,row in new.items():
        unit=key[:-1];path=key[-1];m=row['metrics'];s=path[:2]
        assert path in PATHS and 0<=m['H_macro']<=1
        a,b=m['old_macro_accuracy'],m['new_macro_accuracy']
        assert abs(m['H_macro']-(2*a*b/(a+b) if a+b else 0))<1e-12
        raw=old[unit+(s+'Q0',)];restored=old[unit+(s+'Q1',)]
        assert m['count']==raw['metrics']['count']==restored['metrics']['count']
        assert m['class_counts']==raw['metrics']['class_counts']==restored['metrics']['class_counts']
        for base in (raw,restored):
            if 'split_id' in row:assert row['split_id']==base['split_id'] and row['registered_classes']==base['registered_classes']
        groups.setdefault((row['k'],row['new_count'],path),[]).append((row,raw,restored))
    result=[]
    for (k,n,path),records in sorted(groups.items()):
        assert len(records)==expected_units
        mean=lambda f:statistics.fmean(f(*r) for r in records)
        h=100*mean(lambda r,a,b:r['metrics']['H_macro'])
        old_raw=100*mean(lambda r,a,b:a['metrics']['H_macro']);old_g=100*mean(lambda r,a,b:b['metrics']['H_macro'])
        match=old_raw if path[-1]=='0' else old_g if path[-1]=='1' else None
        delta=h-match if match is not None else None
        raw_delta=h-old_raw;g_delta=h-old_g
        paired=[100*(r['metrics']['H_macro']-(a if path[-1]=='0' else b)['metrics']['H_macro']) for r,a,b in records] if match is not None else []
        if paired:assert abs(statistics.fmean(paired)-delta)<1e-10
        result.append(dict(cohort=cohort,K=k,N=n,path=path,support='raw' if path[1]=='0' else 'DURA',
            query=('raw','G-only','fusion')[int(path[-1])],units=len(records),LocalRidge_H_pct=match,R42_H_pct=h,
            delta_pp=delta,matched_input=match is not None,at_least_5pp=(delta>=5 if delta is not None else None),
            original_raw_H_pct=old_raw,original_G_H_pct=old_g,delta_vs_original_raw_pp=raw_delta,
            delta_vs_original_G_pp=g_delta,cross_raw_at_least5=raw_delta>=5,cross_G_at_least5=g_delta>=5,
            unit_at_least5=sum(v>=5 for v in paired) if paired else None))
    return result


def flag(value):return '✓' if value>=5 else '×'


def pure_cell(row):return f"{row['LocalRidge_H_pct']:.2f}→{row['R42_H_pct']:.2f}（{row['delta_pp']:+.2f} {flag(row['delta_pp'])}）"


def fusion_cell(row):
    a=row['delta_vs_original_raw_pp'];b=row['delta_vs_original_G_pp']
    return f"{row['R42_H_pct']:.2f}；{a:+.2f} {flag(a)}／{b:+.2f} {flag(b)}"


def render_markdown(summary):
    full=summary['main'];pilot=summary['pilot'];p=summary['progress']
    lines=['# R42 已评分结果与当前进度','',f"核对时间：{summary['observed_at_HKT']}。以下宏 H 为 C 阶段逐任务宏 H 的等权平均，H 单位为 %，差值单位为百分点。",'',
        f"R42：完整预测 {p['r42_complete']}/{p['r42_total']} 单元，另 {p['r42_active']} 单元运行；{p['r42_scored']} 个本轮单元已评分。MoPC：{p['mopc_complete_jobs']}/{p['mopc_total_jobs']} 个任务闭合。九原生方法评分：{p['native_complete']}/{p['native_total']}，其中复用 {p['native_reused']}、新评分 {p['native_new']}。",'',
        '新增完整预测按既定全矩阵冻结屏障等待独立评分，没有提前连接新 query 真值。下表仅复用此前已完成的评分；不改变方法或正在运行的队列。','',
        '## 完整主工作点：K=5，N=20','',
        '范围：3 个物理池 × 7 RX × 4 模型种子 × 8 MIX，共 672 个任务单元／路径。达标指该格平均 ΔH≥5，不表示每一个任务单元都≥5。','',
        '| support | query | 原版 H | R42 H | ΔH | 平均≥5 | 单元≥5／672 |','|---|---|---:|---:|---:|---|---:|']
    for r in full:
        if r['matched_input']:lines.append(f"| {r['support']} | {r['query']} | {r['LocalRidge_H_pct']:.2f} | {r['R42_H_pct']:.2f} | {r['delta_pp']:+.2f} | {flag(r['delta_pp'])} | {r['unit_at_least5']}/672 |")
    lines+=['','原版没有融合路径，以下两列是跨输入参考差，不能标为同输入验收：','',
        '| support | 融合 H | 原 raw H | Δ对原 raw | ≥5 | 原 G-only H | Δ对原 G-only | ≥5 |','|---|---:|---:|---:|---|---:|---:|---|']
    for r in full:
        if not r['matched_input']:
            a=r['delta_vs_original_raw_pp'];b=r['delta_vs_original_G_pp']
            lines.append(f"| {r['support']} | {r['R42_H_pct']:.2f} | {r['original_raw_H_pct']:.2f} | {a:+.2f} | {flag(a)} | {r['original_G_H_pct']:.2f} | {b:+.2f} | {flag(b)} |")
    lines+=['','## 全 K／N 已评分先导','',
        '仅 P01／MIX01／RX1-1／模型 seed2026092701，**每格只有 1 个任务单元**。它不是当前完整 672 单元／格的结果，不能用来宣布全矩阵达标。K5/N20 在此处仍保留同一个先导单元，与上方全672均值分开。','',
        'raw、G-only 单元格为“原版 H→R42 H（ΔH，✓≥5／×<5）”；融合列为“R42 H；Δ对原 raw／Δ对原 G-only”，均为跨输入参考。']
    lookup={(r['K'],r['N'],r['path']):r for r in pilot}
    for s,label in ((0,'raw'),(1,'DURA')):
        lines+=['',f'### support={label}','', '| K | N | query raw：原→R42（Δ） | query G-only：原→R42（Δ） | 融合 H；Δ原raw／Δ原G |','|---:|---:|---|---|---|']
        for k in (1,5,10,20):
            for n in (2,5,10,20):lines.append(f"| {k} | {n} | {pure_cell(lookup[k,n,f'S{s}Q0'])} | {pure_cell(lookup[k,n,f'S{s}Q1'])} | {fusion_cell(lookup[k,n,f'S{s}Q2'])} |")
    failed=[r for r in pilot if r['matched_input'] and not r['at_least_5pp']]
    lines+=['',f"先导同输入路径共64格：{64-len(failed)}格≥5、{len(failed)}格<5。未达标格："+', '.join(f"K{r['K']}/N{r['N']}/{r['path']} ({r['delta_pp']:+.2f})" for r in failed)+'。','',
        '来源：已闭合的 R42 main K5N20（20261031-phase2-milkr-r42-main-m4-r01）、全KN先导（20261031-phase2-milkr-adaptres-s2026092701-r41）及各自匹配原版 LocalRidge 独立评分。新恢复run为20261101-phase2-r42-group1-m4-r02，其全矩阵尚在运行。融合无匹配原版，统计范围不同不合并平均；不作独立盲测声明。']
    return '\n'.join(lines)+'\n'


def run(args):
    main=aggregate(load(args.main),'main_full_672',672);pilot=aggregate(load(args.pilot),'pilot_one_unit',1)
    assert len(main)==6 and {(r['K'],r['N']) for r in main}=={(5,20)} and len(pilot)==96
    published={r['path']:r for r in load(args.main_summary)['summary']}
    for r in main:
        assert abs(r['R42_H_pct']-100*published[r['path']]['H'])<1e-10
        if r['matched_input']:assert abs(r['delta_pp']-published[r['path']]['matched_gain_pp'])<1e-10
    status=load(args.status);states=status['runs'];r=states['r42']['states'];m=states['mopc']['states']['pipeline_state'];n=states['native']['states']['score_state']
    progress=dict(r42_complete=r['worker_state']['completed'],r42_total=r['worker_state']['total'],r42_active=len(r['worker_state'].get('active',[])),r42_scored=r.get('score_state',{}).get('completed',0),
        mopc_complete_jobs=m['counts']['COMPLETE'],mopc_total_jobs=sum(m['counts'].values()),native_complete=n['completed'],native_total=n['total'],native_reused=n['reused_jobs'],native_new=n['new_score_jobs'])
    zone=dt.timezone(dt.timedelta(hours=8));summary=dict(observed_at_HKT=dt.datetime.fromtimestamp(status['observed_at'],zone).isoformat(),progress=progress,main=main,pilot=pilot,
        source_files=dict(main=args.main,pilot=args.pilot,main_summary=args.main_summary,status=args.status),new_truth_or_IQ_reads=False)
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    fields=list(main[0]);
    with (out/'results.csv').open('x',encoding='utf-8-sig',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=fields);writer.writeheader();writer.writerows(main+pilot)
    md=render_markdown(summary);(out/'report.md').write_text(md,encoding='utf-8')
    display=[('cohort','样本范围'),('K','K'),('N','N'),('support','support'),('query','query'),('units','单元数'),('LocalRidge_H_pct','原版 H（同输入）'),('R42_H_pct','R42 H'),('delta_pp','同输入 Δ'),('at_least_5pp','同输入 ≥5'),('delta_vs_original_raw_pp','融合对原 raw Δ'),('delta_vs_original_G_pp','融合对原 G Δ')]
    def shown(r,key):
        value=r.get(key)
        if value is None or key.startswith('delta_vs_') and r['matched_input']:return '—'
        if key=='cohort':return '主对比' if value=='main_full_672' else '单元先导'
        if isinstance(value,bool):return '✓' if value else '×'
        if key.startswith('delta_vs_'):return f'{value:+.2f} {flag(value)}'
        if isinstance(value,float):return f'{value:+.2f}' if 'delta' in key else f'{value:.2f}'
        return str(value)
    rows=[]
    for r in main+pilot:
        attributes=' '.join('data-'+k+'="'+html.escape(str(r[k]),quote=True)+'"' for k in ('cohort','K','N','support','query'))
        rows.append('<tr '+attributes+'>'+''.join('<td>'+html.escape(shown(r,k))+'</td>' for k,_ in display)+'</tr>')
    selectors=[]
    for key,label in [('cohort','范围'),('K','K'),('N','N'),('support','support'),('query','query')]:
        values=sorted({str(r[key]) for r in main+pilot})
        options='<option value="">全部</option>'+''.join('<option value="'+html.escape(v,quote=True)+'">'+html.escape({'main_full_672':'主对比（672单元）','pilot_one_unit':'先导（1单元）'}.get(v,v))+'</option>' for v in values)
        selectors.append('<label>'+label+' <select data-key="'+key.lower()+'">'+options+'</select></label>')
    document='<!doctype html><html lang="zh"><meta charset="utf-8"><title>R42 已评分结果</title><style>body{font-family:Arial,sans-serif;padding:24px;color:#192b3b}table{border-collapse:collapse;font-size:14px;white-space:nowrap}th,td{border:1px solid #ddd;padding:9px}th{position:sticky;top:0;background:#edf3f8}label{display:inline-block;margin:0 16px 12px 0}select{padding:7px}tr:hover{background:#f2f7ff}</style><h1>R42 已评分结果</h1><p>主对比：K5/N20，每路径672单元；全16KN先导：每格仅1单元。两种范围分开，不合并平均。H单位为%，Δ为百分点。</p><p>融合没有同输入原版；末两列是跨输入参考差。同输入✓表示平均Δ≥5，×表示小于5。Δ按未舍入值计算。完整精度见配套CSV。</p><p>'+''.join(selectors)+'</p><div style="overflow:auto"><table><thead><tr>'+''.join('<th>'+html.escape(label)+'</th>' for _,label in display)+'</tr></thead><tbody>'+''.join(rows)+'</tbody></table></div><script>let selects=[...document.querySelectorAll("select")];function filter(){document.querySelectorAll("tbody tr").forEach(r=>r.hidden=!selects.every(s=>!s.value||r.dataset[s.dataset.key]===s.value))}selects.forEach(s=>s.onchange=filter)</script></html>'
    (out/'results.html').write_text(document,encoding='utf-8')
    print(json.dumps(dict(output=str(out),rows=len(main)+len(pilot),main_matched_pass=sum(r['at_least_5pp'] is True for r in main),pilot_matched_pass=sum(r['at_least_5pp'] is True for r in pilot),pilot_matched_total=64,progress=progress),ensure_ascii=False))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('main','pilot','main-summary','status','output'):p.add_argument('--'+name,required=True)
    run(p.parse_args())
