"""Paper assets from already verified practical-residual aggregates; no model/truth IO."""
import argparse
import csv
import gc
import json
from pathlib import Path
import shutil

ROOT = Path('E:/type10-7')
RUN = '20261023-phase1-practical-residual-full2x2-m600-r01'
SOURCE = ROOT/'local_artifacts/external_practical_residual_2x2_design_20261023/full_analysis_20261024'
OUT = ROOT/'exports/unified_figures/20261024-practical-residual-paper-r03'
OLD = ROOT/'exports/unified_figures/20261023-pamf-complete-results-r02'
NAMES = {'riei':'RIEI', 'drift':'DRIFT-v2', 'ibrff':'IB-RFF', 'sdg':'SDG', 'pamf':'PAMF'}
METHODS = [('riei','D'),('drift','D'),('ibrff','D'),('sdg','D'),('pamf','C'),('pamf','native_T1')]
LABELS = ['RIEI Full + PL','DRIFT-v2 Full + PL','IB-RFF Full + PL','SDG Full + PL','PAMF supervised (C)','PAMF native T1']
COLORS = ['#52769C','#8B6B99','#C18D53','#639A92','#A6B9CC','#254C78']
TASKS = {'cross_date':'跨日：源 RX、未见 D0', 'cross_rx':'跨接收机：目标 RX、D1–D3', 'cross_both':'跨日且跨接收机：目标 RX、D0'}
TASK_EN = {'cross_date':'Unseen date', 'cross_rx':'Unseen receivers', 'cross_both':'Unseen date and receivers'}

def read(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))

def write(path, rows):
    with path.open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

def route(cell):
    return 'DURA-LU' if cell in ('B','D','native_T1') else 'DURA-L'

def build(raster=False):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    from PIL import Image
    from pypdf import PdfReader, PdfWriter
    from matplotlib import font_manager
    plt.rcParams.update({'font.family':'Times New Roman','font.size':8,'axes.labelsize':8,
        'xtick.labelsize':7,'ytick.labelsize':7,'legend.fontsize':6.5,'svg.fonttype':'none',
        'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.6,
        'savefig.facecolor':'white'})
    assert 'times' in font_manager.findfont('Times New Roman').lower()
    for d in ('pdf','svg','preview','png','tiff','tables','source_data','reproduce','historical_PAMF','IEEE_experiments'):
        (OUT/d).mkdir(parents=True, exist_ok=True)
    rows = read(SOURCE/'summary_overall.csv')
    lookup = {(r['protocol'],r['family'],r['cell'],r['test'],r['readout'],r['view'],r['metric']):r for r in rows}
    assert len(lookup)==len(rows)==20064
    def get(p,f,c,t='cross_rx',rd=None,v='six_practical_mean',m='accuracy'):
        return lookup[p,f,c,t,rd or route(c),v,m]
    def value(r): return float(r['mean'])
    def fmt(r): return f"{float(r['mean']):.2f} ± {float(r['sd']):.2f}"
    figures=[]
    def save(fig,slug,description,caption,role='main',data=None):
        number=len(figures)+1; stem=f'Fig{number:02d}_{slug}'
        assert len(fig.axes)==1
        fig.canvas.draw()
        # All artist extents must remain on canvas, allowing the outward tick stroke.
        ren=fig.canvas.get_renderer(); box=fig.bbox
        for artist in fig.findobj(matplotlib.text.Text):
            if not artist.get_visible() or not artist.get_text(): continue
            b=artist.get_window_extent(ren)
            assert b.x0 >= -2 and b.y0 >= -2 and b.x1 <= box.width+2 and b.y1 <= box.height+2, (stem,artist.get_text(),b)
        fig.savefig(OUT/'pdf'/f'{stem}.pdf')
        fig.savefig(OUT/'svg'/f'{stem}.svg')
        fig.savefig(OUT/'preview'/f'{stem}.png',dpi=180)
        pixels=None
        if raster:
            fig.savefig(OUT/'png'/f'{stem}.png',dpi=4800)
            Image.MAX_IMAGE_PIXELS=None
            with Image.open(OUT/'png'/f'{stem}.png') as im:
                pixels=list(im.size)
                im.save(OUT/'tiff'/f'{stem}.tiff',compression='tiff_lzw',dpi=(4800,4800))
        if data: write(OUT/'tables'/f'{stem}.csv',data)
        figures.append(dict(number=number,stem=stem,description_zh=description,caption_en=caption,
             role=role,font='Times New Roman',dpi=4800 if raster else None,pixels=pixels,
             one_chart=True,no_inside_title=True,no_footnotes=True,width_inches=fig.get_figwidth(),height_inches=fig.get_figheight()))
        plt.close(fig); gc.collect(); print(stem,flush=True)
    common='Three F seeds; mean ± sample SD, conditional on one fixed G per contract and role. Six scenes are averaged within each seed. '
    for t in TASKS:
        for restored in (False,True):
            rec=[get('F0',f,c,t,route(c) if restored else 'raw') for f,c in METHODS]
            fig,ax=plt.subplots(figsize=(3.5,2.6)); fig.subplots_adjust(left=.37,right=.96,bottom=.17,top=.97)
            y=np.arange(6); ax.barh(y,[value(r) for r in rec],xerr=[float(r['sd']) for r in rec],color=COLORS,height=.60,error_kw={'lw':.65,'capsize':1.5})
            ax.set(yticks=y,yticklabels=LABELS,xlim=(0,100),xlabel='Six-channel mean accuracy (%)'); ax.invert_yaxis(); ax.set_axisbelow(True); ax.grid(axis='x',alpha=.15)
            for i,r in enumerate(rec): ax.text(value(r)+float(r['sd'])+2,i,f'{value(r):.2f}',va='center',fontsize=6.6)
            save(fig,t+('_restored' if restored else '_raw'),TASKS[t]+('，对应 DURA 恢复后' if restored else '，直接识别'),
                f'F0, {TASK_EN[t].lower()}, '+('matching DURA + F' if restored else 'direct F')+'. External methods use fixed cell D; PAMF supervised uses C and native PAMF uses T1. '+common, data=rec)
    for task in ('cross_rx','cross_both'):
        for restored in (False,True):
            fig,ax=plt.subplots(figsize=(3.5,3.0));fig.subplots_adjust(left=.17,right=.97,bottom=.31,top=.97)
            records=[]
            for i,(f,c) in enumerate(METHODS):
                rr=[get(f'm{n}-natural',f,c,task,route(c) if restored else 'raw') for n in (1,2,3,4,6)]; records.extend(rr)
                ax.errorbar([1,2,3,4,6],[value(r) for r in rr],yerr=[float(r['sd']) for r in rr],color=COLORS[i],marker=['o','s','^','v','D','P'][i],ms=3,lw=.9,capsize=1,label=LABELS[i])
            ax.set(xlabel='Number of source receivers',ylabel='Six-channel mean accuracy (%)',xticks=[1,2,3,4,5,6],xticklabels=['1','2','3','4','5\nN/T','6'],ylim=(0,100));ax.grid(alpha=.14)
            fig.legend(loc='lower center',ncol=2,bbox_to_anchor=(.53,.005),frameon=False,columnspacing=1,handlelength=1.6)
            save(fig,'source_'+task+('_restored' if restored else '_raw'),TASKS[task]+'的自然预算源数量曲线'+('（恢复后）' if restored else '（直接识别）'),
                 'Natural source budgets, same six target RX, m=1,2,3,4,6. m=5 was not tested; F0 is not inserted. '+common,data=records)
    for protocolset,title,short in [(['label01','label05','F0'],'Label availability','labels'),(['F0','F1_rx_repartition_r01','F2_rx_repartition_r01'],'Receiver partition','receivers')]:
        for restored in (False,True):
            fig,ax=plt.subplots(figsize=(3.5,3));fig.subplots_adjust(left=.17,right=.97,bottom=.31,top=.97)
            records=[]
            for i,(f,c) in enumerate(METHODS):
                rr=[get(p,f,c,rd=route(c) if restored else 'raw') for p in protocolset];records.extend(rr)
                ax.errorbar(range(3),[value(r) for r in rr],yerr=[float(r['sd']) for r in rr],color=COLORS[i],marker=['o','s','^','v','D','P'][i],ms=3,lw=.9,capsize=1,label=LABELS[i])
            ax.set(xticks=range(3),xticklabels=['1%','5%','10%'] if short=='labels' else ['F0','F1','F2'],xlabel=title,ylabel='Six-channel mean accuracy (%)',ylim=(0,100));ax.grid(alpha=.14)
            fig.legend(loc='lower center',ncol=2,bbox_to_anchor=(.53,.005),frameon=False,columnspacing=1,handlelength=1.6)
            save(fig,short+('_restored' if restored else '_raw'),('标注比例' if short=='labels' else '接收机划分')+'的跨接收机结果'+('（恢复后）' if restored else '（直接识别）'),
                 'Cross-RX, '+title.lower()+'. Each condition is reported separately, without target-based cell selection. '+common,data=records)
    for restored in (False,True):
        rec=[get('ManyRx',f,c,rd=route(c) if restored else 'raw') for f,c in METHODS]
        fig,ax=plt.subplots(figsize=(3.5,2.6));fig.subplots_adjust(left=.37,right=.96,bottom=.17,top=.97)
        ax.barh(range(6),[value(r) for r in rec],xerr=[float(r['sd']) for r in rec],color=COLORS,height=.60,error_kw={'lw':.65,'capsize':1.5})
        ax.set(yticks=range(6),yticklabels=LABELS,xlim=(0,100),xlabel='Six-channel mean accuracy (%)');ax.invert_yaxis();ax.grid(axis='x',alpha=.14)
        for i,r in enumerate(rec): ax.text(value(r)+float(r['sd'])+2,i,f'{value(r):.2f}',va='center',fontsize=6.6)
        save(fig,'ManyRx'+('_restored' if restored else '_raw'),'ManyRx 十类跨接收机比较'+('（恢复后）' if restored else '（直接识别）'),'ManyRx ten-class cross-RX evaluation; class set and physical population differ from ManySig. '+common,data=rec)
    for fam in ('riei','drift','ibrff','sdg'):
        records=[]; fig,ax=plt.subplots(figsize=(3.5,2.7));fig.subplots_adjust(left=.18,right=.97,bottom=.24,top=.97)
        for restored,color,marker,label in [(False,'#B68E68','o','Direct F'),(True,'#345F84','s','Matching DURA + F')]:
            rr=[get('F0',fam,c,rd=route(c) if restored else 'raw') for c in 'ABCD'];records.extend(rr)
            ax.errorbar(range(4),[value(r) for r in rr],yerr=[float(r['sd']) for r in rr],color=color,marker=marker,ms=4,lw=1,capsize=2,label=label)
        ax.set(xticks=range(4),xticklabels=['Off / Off','Off / PL','On / Off','On / PL'],xlabel='Channel augmentation / pseudo-labeling',ylabel='Six-channel mean accuracy (%)',ylim=(0,100));ax.grid(alpha=.14)
        fig.legend(loc='lower center',ncol=2,bbox_to_anchor=(.54,.005),frameon=False)
        save(fig,'four_cells_'+fam,NAMES[fam]+' 的四组合及恢复前后实际准确率',
             NAMES[fam]+', F0 cross-RX: A/B/C/D are channel off/PL off, off/on, on/off and on/on. A/C use G-L; B/D use G-LU. Restored PL contrasts also change G source permissions. '+common,role='supplementary',data=records)
    views=['clean','practical_high','practical_mid','practical_low_suburban','practical_high_urban','practical_mid_urban','practical_low_urban']
    for restored in (False,True):
        fig,ax=plt.subplots(figsize=(3.5,3.05));fig.subplots_adjust(left=.17,right=.97,bottom=.32,top=.97);records=[]
        for i,(f,c) in enumerate(METHODS):
            rr=[get('F0',f,c,rd=route(c) if restored else 'raw',v=v) for v in views];records.extend(rr)
            ax.errorbar(range(7),[value(r) for r in rr],yerr=[float(r['sd']) for r in rr],color=COLORS[i],marker=['o','s','^','v','D','P'][i],ms=2.5,lw=.9,capsize=1,label=LABELS[i])
        ax.set(xticks=range(7),xticklabels=['Clean','Sub.\nhigh','Sub.\nmid','Sub.\nlow','Urban\nhigh','Urban\nmid','Urban\nlow'],ylabel='Accuracy (%)',ylim=(0,100));ax.grid(alpha=.14)
        fig.legend(loc='lower center',ncol=2,bbox_to_anchor=(.53,.005),frameon=False,columnspacing=1,handlelength=1.6)
        save(fig,'F0_seven_views'+('_restored' if restored else '_raw'),'F0 跨接收机的 clean 与六种 practical 场景'+('（恢复后）' if restored else '（直接识别）'),
             'F0 cross-RX, clean and all six practical conditions, including those not used for training augmentation. Fixed D external recipes, PAMF supervised C and native T1. '+common,data=records)
    # Preserve useful historical PAMF evidence separately, with its original population.
    oldfile=OLD/'figure_manifest.json'
    if not oldfile.exists():oldfile=OLD/'manifest.json'
    if not oldfile.exists():oldfile=OUT/'historical_PAMF/manifest.json'
    oldmanifest=json.loads(oldfile.read_text(encoding='utf-8'))
    retained=[]
    for rec in oldmanifest['figures']:
        if rec['number'] not in (13,14,18,19,20,23,32,36,37): continue
        for ext,sub in [('pdf','pdf'),('svg','svg'),('png','preview')]:
            src=OLD/sub/(rec['stem']+'.'+ext)
            if not src.exists():src=oldfile.parent/(rec['stem']+'.'+ext)
            dest=OUT/'historical_PAMF'/(rec['stem']+'.'+ext)
            if src.resolve()!=dest.resolve():shutil.copy2(src,dest)
        retained.append(rec)
    (OUT/'historical_PAMF'/'manifest.json').write_text(json.dumps({'status':'VALID_HISTORICAL_PAMF_ONLY','original_population':'F0 all target dates, 168000 packets/view; not the new 126000 cross-RX population','figures':retained},ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    for name in ('summary_overall.csv','factorial_raw_overall.csv','restoration_gain_overall.csv','F_cost.csv','F_stage_cost.csv','G_cost.csv','restored_cache_cost.csv','augmentation_producer_cost.csv','verification.json'):
        shutil.copy2(SOURCE/name,OUT/'source_data'/name)
    tables=[]
    for r in rows:
        if r['view'] not in ('clean','six_practical_mean'):continue
        tables.append(dict(protocol=r['protocol'],method=NAMES[r['family']],cell=r['cell'],population=r['test'],readout=r['readout'],view=r['view'],metric=r['metric'],mean_percent=r['mean'],sample_SD_percent=r['sd'],F_seeds=3))
    write(OUT/'tables'/'all_627_clean_S6.csv',tables)
    main=[]
    for p in ['F0','F1_rx_repartition_r01','F2_rx_repartition_r01','ManyRx','label01','label05']+[f'm{i}-natural' for i in [1,2,3,4,6]]:
        for f,c in METHODS:
            rec=dict(protocol=p,method=LABELS[METHODS.index((f,c))],cell=c)
            for t in TASKS:
                for rd in ('raw',route(c)):
                    for v,short in [('clean','clean'),('six_practical_mean','S6')]:rec[t+'_'+('G' if rd!='raw' else 'raw')+'_'+short]=fmt(get(p,f,c,t,rd,v))
            main.append(rec)
    write(OUT/'tables'/'paper_main_comparison.csv',main)
    figmanifest={'status':'VERIFIED_AGGREGATE_DERIVATION','source_run':RUN,'source_rows':627,'statistics':'mean and sample SD across three F seeds, scenes within seed first','figures':figures,'retained_historical_figures':len(retained)}
    (OUT/'figure_manifest.json').write_text(json.dumps(figmanifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    for role in ('main','supplementary'):
        writer=PdfWriter()
        for x in figures:
            if x['role']==role: writer.append(str(OUT/'pdf'/(x['stem']+'.pdf')))
        writer.write(str(OUT/(role+'_figures.pdf')));writer.close()
    # Tables avoid excessive width: distinct tasks and direct/restored routes get their own tables.
    tex=[r'% Ready-to-include tables; requires booktabs. All entries mean +- sample SD (%).']
    for t in TASKS:
        tex += [r'\begin{table*}[t]\centering\small',r'\setlength{\tabcolsep}{5pt}\renewcommand{\arraystretch}{1.15}',
            r'\caption{F0 '+TASK_EN[t].lower()+r': clean and six-channel mean accuracy (\%). Three F seeds, fixed G per role.}',
            r'\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lrrrr@{}}\toprule',r'Method & Direct clean & Direct S6 & DURA clean & DURA S6\\\midrule']
        for f,c in METHODS:
            vals=[get('F0',f,c,t,rd,v) for rd in ('raw',route(c)) for v in ('clean','six_practical_mean')]
            tex.append(LABELS[METHODS.index((f,c))]+' & '+' & '.join('$'+fmt(x).replace(' ± ',r'\pm')+'$' for x in vals)+r'\\')
        tex += [r'\bottomrule\end{tabular*}\end{table*}','']
    (OUT/'tables'/'paper_tables.tex').write_text('\n'.join(tex)+'\n',encoding='utf-8')
    captions='\n\n'.join('### '+x['stem']+'\n\n'+x['caption_en'] for x in figures)
    (OUT/'figure_captions_en.md').write_text(captions+'\n',encoding='utf-8')
    f0=get('F0','pamf','native_T1');sdg=get('F0','sdg','D');sup=get('F0','pamf','C')
    prose=f'''# Paper-ready experimental results

## Protocol and comparison design

We evaluated four complete comparison methods (RIEI, DRIFT-v2, IB-RFF and SDG) under a preregistered 2×2 design: channel augmentation off/on and ordinary pseudo-labeling off/on. All methods retained their native objectives, classifiers and training budgets. Channel augmentation used the practical ResidualChannel in residual/post-sync/no-equalization mode, with low suburban, mid urban and low urban sampled with probabilities 0.40, 0.35 and 0.25. Application probabilities followed 0.30/0.60/0.80 over the registered training-budget intervals. Ordinary pseudo-labeling used the current model, confidence threshold 0.95 and weight 1. PAMF supervised controls and native semi-supervised T1 were evaluated separately; native T1 was not replaced by ordinary pseudo-labeling.

Training used D1–D3. Three disjoint evaluation populations tested unseen dates (source receivers on D0), unseen receivers (target receivers on D1–D3), and their combination (target receivers on D0). D0 precedes the training dates, so this experiment does not establish future-date extrapolation. Each row was frozen across all populations, raw/matching-DURA routes, clean and six practical scenes before independent scoring. We report three-F-seed means and sample SD, averaging scenes within each seed. These SDs condition on one G per contract and role and are not confidence intervals.

## Main comparison

For F0 cross-receiver generalization, native PAMF-T1 with matching DURA reached {fmt(f0)}% S6 accuracy, compared with {fmt(sdg)}% for SDG Full with both augmentation and ordinary pseudo-labeling. The difference in means was {value(f0)-value(sdg):.2f} percentage points. The corresponding PAMF supervised channel-on system reached {fmt(sup)}%. The external D recipe was fixed in advance rather than selected by target accuracy. All A/B/C/D cells remain available in the complete tables; the highlighted D comparison does not imply that D is optimal for every method.

## Generalization and limitations

Performance varied across receiver partitions. Native PAMF-T1 achieved {fmt(get('F1_rx_repartition_r01','pamf','native_T1'))}% and {fmt(get('F2_rx_repartition_r01','pamf','native_T1'))}% on the F1 and F2 cross-receiver S6 endpoints. On the F2 joint date/receiver shift, T1 reached {fmt(get('F2_rx_repartition_r01','pamf','native_T1','cross_both'))}% whereas the supervised channel-on recipe reached {fmt(get('F2_rx_repartition_r01','pamf','C','cross_both'))}%. Thus, the evidence does not support uniformly positive semi-supervised gains. Source-count results use only m=1,2,3,4,6 under natural budgets and fixed six-target populations; no m=5 result is interpolated, and F0 is not inserted into the curve. ManyRx uses ten classes and is reported separately from six-class ManySig.

Augmentation and pseudo-labeling effects are method- and condition-dependent. All negative effects and the 61 external PL rows with zero accepted pseudo-labels are retained. A/C restoration uses G-L, while B/D and native T1 use G-LU; a restored comparison across PL cells therefore also changes the restorer's source-IQ permissions and is a complete-system comparison. Raw within-method 2×2 contrasts provide the cleaner augmentation/PL attribution. Existing PAMF module and structure studies retain their original all-target-date population and cannot be relabeled as the new cross-receiver endpoint.

## Computation and reporting boundaries

The 600 new F models required 246.37 shared-worker elapsed hours; 27 reused F sources account for 46.26 historical hours. Four new complete G lineages required 18.42 hours. These costs are not exclusive GPU hours or wall-clock completion time. Cached classifier prediction time is not online G+F latency. Full native PAMF acceptance and per-F realized channel counts are unavailable in the compact telemetry and remain missing. Existing dedicated timing and mechanism evidence is preserved separately, without claiming new practical-residual latency measurements.
'''
    (OUT/'manuscript_results_en.md').write_text(prose,encoding='utf-8')
    (OUT/'README_zh.md').write_text('''# practical residual 论文资料包

本包仅使用 20261023-phase1-practical-residual-full2x2-m600-r01 的 627 行完整读出。旧错误增强的外部比较不进入新图表。

- manuscript_results_en.md：可用于论文的英文设置、结果与边界。
- tables/paper_main_comparison.csv：11 契约、三类泛化的主要方法 clean/S6 原始与恢复后分数。
- tables/all_627_clean_S6.csv：所有四组合、全部指标；paper_tables.tex 可直接引入 LaTeX。
- pdf/svg：22 张新独立图；preview：预览；png/tiff：4800 dpi。
- figure_captions_en.md：英文图注；中文逐图说明另见 PDF。
- source_data：原已验证汇总与成本，未重新连接目标真值。
- historical_PAMF：9 张有效历史机制/结构/时延图，原 168000 包口径，不并入新跨 RX 分数。

组合 A/B/C/D 依次为无增强无 PL、无增强有 PL、有增强无 PL、有增强有 PL；均为 Full。主图固定外部 D、PAMF 监督 C、PAMF 原生 T1，绝非按目标结果选最佳组合。A/C 使用 G-L，B/D/T1 使用 G-LU。误差为三 F 种子样本标准差，同 G 条件下变化。

训练 D1–D3；跨日=源 RX 的 D0，跨 RX=目标 RX 的 D1–D3，双跨=目标 RX 的 D0。D0 是较早未见日。自然源数量只测 1/2/3/4/6，不混入五源 F0。所有负值和缺项保留。
''',encoding='utf-8')
    shutil.copy2(Path(__file__),OUT/'reproduce'/Path(__file__).name)
    print(json.dumps({'output':str(OUT),'figures':len(figures),'historical':len(retained),'tables':len(tables),'raster':raster}),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--raster',action='store_true')
    ap.add_argument('--source',type=Path,default=SOURCE);ap.add_argument('--output',type=Path,default=OUT)
    ap.add_argument('--historical',type=Path,default=OLD);args=ap.parse_args()
    SOURCE,OUT,OLD=args.source,args.output,args.historical
    build(args.raster)
