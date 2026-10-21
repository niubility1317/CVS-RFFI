"""Summarize completed final-T1 F0 scores only; never opens IQ/truth or trains."""
import argparse,csv,json,math,statistics
from pathlib import Path
from .io import read,write
VIEWS=('clean','practical_high','practical_mid','practical_low_suburban','practical_high_urban','practical_mid_urban','practical_low_urban')


def summarize(folder,output,paper):
    folder,output,paper=map(Path,(folder,output,paper));output.mkdir(parents=True,exist_ok=True)
    rows=[];pair_rows=[]
    for seed in (2026092701,2026092702,2026092703):
        record=read(folder/(str(seed)+'.json'));s=record['scores'];done=record['complete']
        if s['model_seed']!=seed or not s['truth_last'] or not s['target_feedback_forbidden'] or done['status']!='SCORED_COMPLETE' or done['views']!=list(VIEWS):raise ValueError('Complete three-seed truth-last F0 required')
        actual={}
        for r in s['results']:
            cm=r['confusion'];n=sum(map(sum,cm));correct=sum(cm[i][i] for i in range(6));f1=[]
            for i in range(6):
                denom=sum(cm[i])+sum(row[i] for row in cm);f1.append(2*cm[i][i]/denom if denom else 0.)
            if n!=r['count'] or abs(correct/n-r['accuracy'])>1e-12 or abs(sum(f1)/6-r['macro_f1'])>1e-12:raise ValueError('Confusion-derived metric differs')
            if r['dimension']=='overall':
                if n!=168000 or r['test']!='main':raise ValueError('Different physical target set')
                actual[(r['readout'],r['view'])]=r
                worst=min(v['accuracy'] for v in s['results'] if v['readout']==r['readout'] and v['view']==r['view'] and v['dimension']=='receiver')
                rows.append(dict(seed=seed,readout=r['readout'],view=r['view'],count=n,correct=correct,accuracy=100*r['accuracy'],macro_f1=100*r['macro_f1'],worst_rx=100*worst))
        if set(actual)!={(name,view) for name in ('raw','DURA-LU') for view in VIEWS}:raise ValueError('Incomplete raw/G view matrix')
        for r in s['paired_raw']:
            if r['dimension']!='overall':continue
            expected=actual[('DURA-LU',r['view'])]['accuracy']-actual[('raw',r['view'])]['accuracy']
            if r['net']!=r['repairs']-r['damage'] or abs(r['net']/r['count']-expected)>1e-12:raise ValueError('Paired net correction differs')
            pair_rows.append(dict(seed=seed,**r,delta_pp=100*expected))
    summary=[]
    for name in ('raw','DURA-LU'):
        for view in (*VIEWS,'six_view_mean'):
            per=[]
            for seed in (2026092701,2026092702,2026092703):
                selected=[r for r in rows if r['seed']==seed and r['readout']==name and (r['view'] in VIEWS[1:] if view=='six_view_mean' else r['view']==view)]
                per.append({key:statistics.mean(r[key] for r in selected) for key in ('accuracy','macro_f1','worst_rx')})
            summary.append(dict(readout=name,view=view,F_seeds=3,G_seeds=0 if name=='raw' else 1,
                **{key+suffix:fn(v[key] for v in per) for key in ('accuracy','macro_f1','worst_rx') for suffix,fn in (('_mean',statistics.mean),('_sd',statistics.stdev))}))
    for filename,data in [('final_t1_f0_per_seed.csv',rows),('final_t1_f0_summary.csv',summary),('final_t1_f0_paired.csv',pair_rows)]:
        with (output/filename).open('w',encoding='utf-8',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(data[0]));writer.writeheader();writer.writerows(data)
    text=['# Final T1-E50＋DURA：F0 三种子七景结果','',
        '已完成子矩阵：最终T1 13L＋37U，三个F seed，共用一个固定DURA v1.0；源5RX、目标7RX、每景168000物理样本，clean＋六practical。不是整批17契约实验完成。全部预测后独立truth-last，混淆矩阵与净纠错已复算。','',
        '|场景|T1 raw（%）|T1＋DURA（%）|配对增益（百分点）|','|---|---:|---:|---:|']
    for view in (*VIEWS,'six_view_mean'):
        a=next(r for r in summary if r['readout']=='raw' and r['view']==view);b=next(r for r in summary if r['readout']=='DURA-LU' and r['view']==view)
        deltas=[(statistics.mean(p['delta_pp'] for p in pair_rows if p['seed']==seed and p['view'] in VIEWS[1:]) if view=='six_view_mean' else next(p['delta_pp'] for p in pair_rows if p['seed']==seed and p['view']==view)) for seed in (2026092701,2026092702,2026092703)]
        text.append('|%s|%.4f±%.4f|%.4f±%.4f|%+.4f±%.4f|'%(view,a['accuracy_mean'],a['accuracy_sd'],b['accuracy_mean'],b['accuracy_sd'],statistics.mean(deltas),statistics.stdev(deltas)))
    text+=['','SD为三个F seed在单个固定G条件下的样本SD；六景均值先在每个seed内算，再跨seed汇总。旧无As33L17U＋DURA另列历史，不混合平均或替换默认。其余配置、F/G机制和正式时延继续运行，结果不反馈研发。']
    (output/'final_T1_F0_results.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
    table=[r'\begin{table*}[t]\centering\footnotesize',r'\caption{Final T1-E50 (13L/37U), F0 five-source/seven-target RX, 168000 packets per view. Three F seeds share one fixed DURA; mean $\pm$ sample SD, accuracy in percent.}',r'\label{tab:final-t1-f0}',r'\begin{tabular}{lrrrrrrrr}\toprule',r'Readout & Clean & S-high & S-mid & S-low & U-high & U-mid & U-low & S6\\\midrule']
    for name,label in [('raw','Final T1'),('DURA-LU','Final T1+DURA')]:
        values=[next(r for r in summary if r['readout']==name and r['view']==view) for view in (*VIEWS,'six_view_mean')]
        table.append(label+' & '+' & '.join('$%.2f\\!\\pm\\!%.2f$'%(r['accuracy_mean'],r['accuracy_sd']) for r in values)+r' \\')
    table += [r'\bottomrule\end{tabular}\end{table*}']
    (paper/'tables/final_t1_f0.tex').write_text('\n'.join(table)+'\n',encoding='utf-8')
    write(output/'final_t1_f0_verification.json',dict(status='ANALYZED_F0_SUBMATRIX',run_id='20261021-phase1-T1-DURA-comparison-m153-r01',F_seeds=3,G_seeds=1,physical_samples_per_view=168000,raw_and_G_view_cells=42,all_confusions_recomputed=True,all_paired_nets_recomputed=True,target_feedback=False))
    print('\n'.join(text[:13]))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--output',required=True);p.add_argument('--paper',required=True)
    a=p.parse_args();summarize(a.input,a.output,a.paper)
