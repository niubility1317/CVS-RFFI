"""Reproduce publication figures from the frozen standard-CE clean benchmark.

No model inference, truth access, training or experiment selection is performed.
Usage: python plot_standard_ce_clean.py --results PATH --output NEW_DIRECTORY
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans']
plt.rcParams['svg.fonttype'] = 'none'
import numpy as np
from matplotlib.lines import Line2D
from pypdf import PdfReader, PdfWriter
from PIL import Image

plt.rcParams.update({'pdf.fonttype': 42, 'font.size': 7.5, 'axes.titlesize': 8,
    'axes.labelsize': 8, 'axes.linewidth': .65, 'axes.spines.top': False,
    'axes.spines.right': False, 'xtick.labelsize': 7, 'ytick.labelsize': 7.5,
    'legend.fontsize': 7, 'legend.frameon': False, 'savefig.facecolor': 'white'})

RUN = '20261010-phase1-standard-ce-baselines-manysig-m56-r01'
WIDTH = 183/25.4
NAMES = {
    'residual_fusion': 'CVS', 'native': 'Original CVS',
    'resnet50': 'ResNet-50-1D', 'cvresnet18_1d': 'Complex ResNet-18-1D',
    'resnet34': 'ResNet-34-1D', 'resnet18': 'ResNet-18-1D',
    'densenet121': 'DenseNet-121-1D', 'vgg16_bn': 'VGG-16-BN-1D',
    'bilstm3': 'BiLSTM (3 layers)', 'resnet18_2d': 'ResNet-18-2D',
    'convnext_tiny': 'ConvNeXt-Tiny-1D', 'resnet50_2d': 'ResNet-50-2D',
    'cvresnet18_2d': 'Complex ResNet-18-2D', 'transformer6': 'Transformer (6 layers)',
}
COLORS = {'cvs': '#BB6C42', 'real1d': '#507398', 'real2d': '#7884B4',
          'complex': '#87709B', 'sequence': '#648D89', 'seed': '#A0A5AA'}
SEEDS = [2026092701, 2026092702, 2026092703, 2026092704]
SEED_OFFSETS = np.array([-.13, -.043, .043, .13])

def family(v):
    if v in ('native', 'residual_fusion'): return 'cvs'
    if v.startswith('cvresnet'): return 'complex'
    if v.endswith('_2d'): return 'real2d'
    if v in ('bilstm3', 'transformer6'): return 'sequence'
    return 'real1d'

def csvread(path):
    with path.open(encoding='utf-8-sig', newline='') as f: return list(csv.DictReader(f))

def csvwrite(path, rows):
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)

def panel(ax, letter, title):
    ax.set_title(title, loc='left', pad=12)
    ax.text(-.035, 1.055, letter, transform=ax.transAxes, weight='bold', fontsize=10,
            ha='right', va='bottom')

def grid(ax, axis='x'):
    ax.set_axisbelow(True);ax.grid(axis=axis, color='#E6E8EA', linewidth=.5)

def foot(fig, text, y=.023):
    fig.text(.5, y, text, ha='center', va='bottom', fontsize=7, color='#555555')

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--resume-render',action='store_true',help='Repair an incomplete export of the same snapshot')
    args = parser.parse_args()
    source = args.results.resolve();out = args.output.resolve()
    if out.exists() and not args.resume_render: raise FileExistsError(f'Use a new output directory: {out}')
    raw = json.loads((source/'verified_results.json').read_text(encoding='utf-8'))
    if out.exists() and args.resume_render:
        previous=json.loads((out/'source_data/verified_results.json').read_text(encoding='utf-8'))
        if previous.get('snapshot_hkt')!=raw.get('snapshot_hkt') or previous['run_id']!=raw['run_id']:raise ValueError('Different existing export snapshot')
    progress_mode='snapshot_hkt' in raw
    if (raw['run_id'] != RUN and not progress_mode) or raw['status'] != 'VERIFIED': raise ValueError('Wrong run or unverified results')
    rows = raw['rows']
    if not progress_mode:assert len(rows)==56
    seed_map={v:sorted(r['row']['model_seed'] for r in rows if r['row']['variant']==v) for v in NAMES}
    present=[v for v in NAMES if seed_map[v]]
    if progress_mode:assert 'native' not in present and len(present)==13
    counts=[len(seed_map[v]) for v in present]
    n_context=f'n = {min(counts)}–{max(counts)} completed model seeds' if min(counts)!=max(counts) else f'n = {counts[0]} completed model seeds'
    labels={v:NAMES[v]+(f' (n={len(seed_map[v])})' if progress_mode else '') for v in present}
    def seed_offsets(v):return np.linspace(-.15,.15,len(seed_map[v]))
    metric = {}
    for r in rows:
        v=r['row']['variant'];seed=r['row']['model_seed']
        assert v in NAMES and (seed in SEEDS or progress_mode and 2026092705<=seed<=2026092708)
        for s in r['scores']['results']:
            metric[v,seed,s['dimension'],s['stratum']]=s
        overall=metric[v,seed,'overall','ALL'];cm=np.asarray(overall['confusion'])
        assert cm.shape==(6,6) and np.all(cm.sum(axis=1)==28000) and cm.sum()==168000
        assert abs(100*cm.trace()/cm.sum()-100*overall['accuracy'])<1e-10
    def values(v, key='accuracy', dim='overall', stratum='ALL'):
        return np.array([100*metric[v,s,dim,stratum][key] for s in seed_map[v]])
    def paired_values(a,b):
        seeds=sorted(set(seed_map[a])&set(seed_map[b]))
        return seeds,np.array([100*(metric[a,s,'overall','ALL']['accuracy']-metric[b,s,'overall','ALL']['accuracy']) for s in seeds])
    order=sorted(present, key=lambda v:-values(v).mean())
    summary={r['variant']:r for r in csvread(source/'network_summary.csv')}
    for v in order:
        assert abs(values(v).mean()-float(summary[v]['accuracy_pct']))<1e-10
        assert abs(values(v).std(ddof=1)-float(summary[v]['accuracy_sd_pp']))<1e-10
        assert int(summary[v]['actual_seed_count'])==len(seed_map[v])
    rx=[s['stratum'] for s in rows[0]['scores']['results'] if s['dimension']=='receiver']
    tx=raw['classes'];assert len(rx)==7 and len(tx)==6
    out.mkdir(parents=True,exist_ok=args.resume_render)
    for sub in ['pdf','svg','png','preview','source_data']: (out/sub).mkdir(exist_ok=args.resume_render)
    for name in ['network_summary.csv','per_seed_metrics.csv','receiver_summary.csv',
                 'transmitter_summary.csv','paired_differences.csv','resource_profiles.csv']:
        shutil.copyfile(source/name,out/'source_data'/name)
    # A self-contained, shareable score source excludes process/environment paths.
    safe_source=dict(run_id=raw['run_id'],status='VERIFIED',classes=tx,
        rows=[dict(row={k:r['row'][k] for k in ('variant','model_seed')},
                   scores=dict(results=r['scores']['results'])) for r in rows])
    if progress_mode:safe_source.update(snapshot_hkt=raw['snapshot_hkt'],progress=raw['progress'],run_states=raw['run_states'])
    (out/'source_data/verified_results.json').write_text(json.dumps(safe_source,ensure_ascii=False)+'\n',encoding='utf-8')
    records=[]
    def save(fig, stem, caption, inputs):
        # Preserve the same physical canvas across exports, without a tight crop.
        fig.canvas.draw()
        canvas=fig.get_window_extent();renderer=fig.canvas.get_renderer()
        clipped=[]
        for text in fig.findobj(matplotlib.text.Text):
            if not text.get_visible() or not text.get_text():continue
            bbox=text.get_window_extent(renderer)
            if bbox.width and bbox.height and (bbox.x0 < -1 or bbox.y0 < -1 or bbox.x1 > canvas.x1+1 or bbox.y1 > canvas.y1+1):
                clipped.append(text.get_text())
        if clipped: raise ValueError(f'{stem}: labels outside canvas: {clipped}')
        fig.savefig(out/'svg'/f'{stem}.svg')
        fig.savefig(out/'pdf'/f'{stem}.pdf')
        fig.savefig(out/'png'/f'{stem}.png',dpi=600)
        fig.savefig(out/'preview'/f'{stem}.png',dpi=160)
        root=ET.parse(out/'svg'/f'{stem}.svg').getroot()
        assert root.findall('.//{http://www.w3.org/2000/svg}text'), 'SVG text was outlined'
        pdf=PdfReader(out/'pdf'/f'{stem}.pdf');assert len(pdf.pages)==1 and len(pdf.pages[0].extract_text())>40
        width=float(pdf.pages[0].mediabox.width)/72*25.4
        assert abs(width-183)<.1
        with Image.open(out/'png'/f'{stem}.png') as im:
            assert abs(im.info['dpi'][0]-600)<1
            dimensions=list(im.size)
        records.append(dict(stem=stem,caption=caption,source_data=inputs,width_mm=183,
                            height_mm=round(fig.get_figheight()*25.4,2),png_pixels=dimensions,
                            formats=['svg','pdf','png'],svg_editable_text=True,pdf_extractable_text=True))
        plt.close(fig)

    # Figure 1: all seeds, means and SD; no zero-truncated categorical bars.
    fig,axes=plt.subplots(1,2,figsize=(WIDTH,5.5),sharey=True)
    fig.subplots_adjust(left=.255,right=.982,bottom=.195,top=.895,wspace=.16)
    for ai,(ax,key,title) in enumerate(zip(axes,['accuracy','macro_f1'],['Clean-test accuracy','Clean-test macro-F1'])):
        for i,v in enumerate(order):
            a=values(v,key);c=COLORS[family(v)]
            if v in ('native','residual_fusion'):ax.axhspan(i-.43,i+.43,color='#F7EDE6',zorder=0)
            ax.scatter(a,i+seed_offsets(v),s=10,color=COLORS['seed'],linewidths=0,zorder=3)
            ax.errorbar(a.mean(),i,xerr=a.std(ddof=1),fmt='D',ms=4,color=c,capsize=2,elinewidth=1.2,zorder=4)
            ax.text(104,i,f'{a.mean():.2f}',fontsize=7,va='center',ha='right',color=c)
        ax.set_xlim(0,107);ax.set_xticks([0,25,50,75,100]);ax.set_ylim(len(order)-.6,-.6)
        ax.set_yticks(range(len(order)),[labels[v] for v in order]);ax.tick_params(axis='y',length=0)
        ax.set_xlabel('Accuracy (%)' if ai==0 else 'Macro-F1 (%)');grid(ax);panel(ax,'ab'[ai],title)
    fig.legend([Line2D([],[],marker='o',ls='',color=COLORS['seed'],ms=3),
                Line2D([],[],marker='D',color='#507398',ms=4)],
               ['Individual model seed','Mean ± sample SD'],loc='lower center',bbox_to_anchor=(.62,.065),ncol=2)
    foot(fig,n_context+'; 168,000 identical packets per model; E200.')
    save(fig,'fig01_overall_performance',
         f'Clean-test accuracy (a) and macro-F1 (b) for {len(order)} architectures. Labels give actual completed seed counts. Grey points show every completed model seed; diamonds and error bars show mean and sample SD (ddof=1). Numeric labels are means. Collapsed seeds are retained. Ordering is for display only.',
         ['network_summary.csv','per_seed_metrics.csv'])

    # Figure 2: fixed 0--100 scale and all receiver cells.
    matrix=np.array([[values(v,dim='receiver',stratum=r).mean() for r in rx] for v in order])
    fig,ax=plt.subplots(figsize=(WIDTH,5.2));fig.subplots_adjust(left=.255,right=.895,bottom=.14,top=.88)
    im=ax.imshow(matrix,cmap='Blues',vmin=0,vmax=100,aspect='auto',interpolation='nearest')
    for i in range(len(order)):
        for j in range(len(rx)):
            ax.text(j,i,f'{matrix[i,j]:.1f}',ha='center',va='center',fontsize=7,
                    color='white' if matrix[i,j]>=63 else '#243449')
    ax.set_xticks(range(len(rx)),rx);ax.set_yticks(range(len(order)),[labels[v] for v in order]);ax.tick_params(length=0)
    ax.set_xlabel('Held-out receiver (physical ID)',labelpad=8)
    for s in ax.spines.values():s.set_visible(False)
    cax=fig.add_axes([.915,.14,.019,.74]);cb=fig.colorbar(im,cax=cax);cb.set_label('Accuracy (%)');cb.set_ticks([0,25,50,75,100])
    panel(ax,'a','Accuracy across held-out receivers')
    foot(fig,'Actual n shown per network; 24,000 packets per receiver and model.')
    save(fig,'fig02_receiver_heatmap',
         'Mean clean-test accuracy across seven held-out receivers, with 24,000 packets per receiver and model. Actual completed seed counts appear beside each network. Colours use a fixed 0–100% scale. Receiver SD is supplied in source data.',
         ['receiver_summary.csv'])

    # Figure 3: points numbered to avoid overlapping architecture labels.
    fig=plt.figure(figsize=(WIDTH,4.8));ax=fig.add_axes([.09,.18,.48,.68]);keyax=fig.add_axes([.615,.20,.36,.64]);keyax.axis('off')
    for i,v in enumerate(order,1):
        a=values(v);p=int(summary[v]['total_parameters'])/1e6;c=COLORS[family(v)]
        marker='D' if family(v)=='cvs' else 's' if family(v)=='complex' else '^' if family(v)=='real2d' else 'o'
        ax.errorbar(p,a.mean(),yerr=a.std(ddof=1),fmt=marker,color=c,ms=6,capsize=2,elinewidth=.9)
        offsets={1:(-8,8),2:(-8,-13),3:(3,8),4:(5,9),5:(-10,-14),6:(-13,6),7:(-9,-16),8:(-13,8),
                 9:(-15,2),10:(5,7),11:(4,8),12:(-17,0),13:(-17,-12),14:(5,-10)}
        dx,dy=offsets[i]
        ax.annotate(str(i),(p,a.mean()),xytext=(dx,dy),textcoords='offset points',fontsize=7,color=c,
                    bbox=dict(boxstyle='round,pad=.12',fc='white',ec='none',alpha=.9))
        keyax.text(0,1-(i-1)/len(order),f'{i:>2}  {labels[v]}',fontsize=7.2,color=c,va='top')
    ax.set_xscale('log');ax.set_xlim(.1,65);ax.set_ylim(0,100);ax.set_yticks([0,20,40,60,80,100])
    ax.set_xticks([.1,1,10,50],['0.1','1','10','50']);ax.set_xlabel('Total parameters (millions; log scale)');ax.set_ylabel('Clean-test accuracy (%)')
    grid(ax,'both');panel(ax,'a','Capacity and clean-test accuracy')
    foot(fig,'Mean ± sample SD; actual n in key. Parameters do not measure inference cost.',.048)
    save(fig,'fig03_parameter_accuracy',
         'Clean-test accuracy versus parameter count on a logarithmic axis in millions. Points and vertical error bars show completed-seed means and sample SD; actual n appears in the key. Parameters do not measure full compute, memory or latency.',
         ['network_summary.csv'])

    # Figure 4: same-seed differences, including negative results.
    pairs=[('resnet18','resnet18_2d','ResNet-18'),('resnet50','resnet50_2d','ResNet-50'),
           ('cvresnet18_1d','cvresnet18_2d','Complex ResNet-18'),
           ('cvresnet18_1d','resnet18','ResNet-18-1D'),('cvresnet18_2d','resnet18_2d','ResNet-18-2D')]
    pairrows=[]
    fig,axes=plt.subplots(1,2,figsize=(WIDTH,3.6));fig.subplots_adjust(left=.195,right=.97,bottom=.26,top=.79,wspace=.72)
    for ai,(ax,selected,title) in enumerate(zip(axes,[pairs[:3],pairs[3:]],['1D minus 2D','Complex minus real'])):
        ax.axvline(0,color='#7D8389',ls='--',lw=.8)
        for i,(a,b,label) in enumerate(selected):
            pair_seeds,d=paired_values(a,b);c=COLORS['real1d'] if ai==0 else COLORS['complex']
            ax.scatter(d,i+np.linspace(-.15,.15,len(d)),s=15,color=COLORS['seed'],zorder=3)
            ax.errorbar(d.mean(),i,xerr=d.std(ddof=1),fmt='D',color=c,ms=4,capsize=2,zorder=4)
            ax.text(.98,i+.20,f'{d.mean():+.2f} pp; n={len(d)}',transform=ax.get_yaxis_transform(),ha='right',va='top',fontsize=7,color=c)
            for seed,value in zip(pair_seeds,d):pairrows.append(dict(comparison=title,first=a,second=b,model_seed=seed,accuracy_difference_pp=float(value)))
        ax.set_yticks(range(len(selected)),[p[2] for p in selected]);ax.set_ylim(len(selected)-.25,-.6);ax.tick_params(axis='y',length=0)
        ax.set_xlim((-3,20) if ai==0 else (-12,8));ax.set_xticks([0,5,10,15,20] if ai==0 else [-10,-5,0,5])
        ax.set_xlabel('Accuracy difference (percentage points)');grid(ax);panel(ax,'ab'[ai],title)
    foot(fig,'Same model seed and test packets; grey points: paired seeds; diamonds: mean ± sample SD.',.10)
    foot(fig,'2D inputs use a deterministic 16 × 16 IQ reshape; input adaptation is part of the comparison.',.042)
    csvwrite(out/'source_data/architecture_paired_differences.csv',pairrows)
    save(fig,'fig04_architecture_paired_differences',
         'Same-seed accuracy differences between 1D and 2D counterparts (a) and complex and real ResNet-18 counterparts (b). Only the intersection of completed seeds is paired, with actual pair count shown. Points and error bars show paired mean and sample SD. The 2D input uses a 16×16 IQ reshape, which is part of the comparison.',
         ['architecture_paired_differences.csv'])

    # Figure 5: current CVS receiver and transmitter performance.
    fig,axes=plt.subplots(1,2,figsize=(WIDTH,4.4));fig.subplots_adjust(left=.095,right=.975,bottom=.21,top=.83,wspace=.28)
    stratarows=[]
    for ai,(ax,dim,strata,title) in enumerate(zip(axes,['receiver','transmitter'],[rx,tx],['Held-out receivers','Transmitter identities'])):
        for i,s in enumerate(strata):
            residual=values('residual_fusion',dim=dim,stratum=s)
            ax.errorbar(residual.mean(),i,xerr=residual.std(ddof=1),fmt='D',ms=4,color=COLORS['cvs'],capsize=2,elinewidth=.8,zorder=3)
            ax.scatter(residual,i+seed_offsets('residual_fusion'),s=10,color=COLORS['seed'],zorder=2)
            ax.text(105,i,f'{residual.mean():.2f}',ha='left',va='center',fontsize=7)
            for seed,value in zip(seed_map['residual_fusion'],residual):
                stratarows.append(dict(dimension=dim,stratum=s,model_seed=seed,accuracy_pct=float(value)))
        ax.text(105,-.65,'Mean (%)',ha='left',fontsize=7,color='#555555')
        ax.set_yticks(range(len(strata)),strata);ax.tick_params(axis='y',length=0);ax.set_ylim(len(strata)-.6,-1)
        ax.set_xlim(0,121);ax.set_xticks([0,25,50,75,100]);ax.set_xlabel('Clean-test accuracy (%)');grid(ax);panel(ax,'ab'[ai],title)
    fig.legend([Line2D([],[],marker='o',ls='',color=COLORS['seed'],ms=3),Line2D([],[],marker='D',ls='',color=COLORS['cvs'],ms=4)],
               ['Individual model seed','CVS mean ± sample SD'],loc='lower center',bbox_to_anchor=(.5,.09),ncol=2)
    foot(fig,f'CVS: n = {len(seed_map["residual_fusion"])} completed seeds; transmitter accuracy is class recall.',.035)
    csvwrite(out/'source_data/cvs_stratum_per_seed.csv',stratarows)
    save(fig,'fig05_cvs_stratum_comparison',
         f'CVS performance across held-out receivers (a) and transmitter identities (b), using {len(seed_map["residual_fusion"])} completed seeds. Grey points show seeds; diamonds and horizontal error bars show mean and sample SD. Each receiver contains 24,000 packets and each transmitter 28,000 packets per model. Transmitter accuracy equals class recall.',
         ['receiver_summary.csv','transmitter_summary.csv','cvs_stratum_per_seed.csv'])

    # Figure 6: mean row-normalized confusion percentages, not a model ensemble.
    fig,axes=plt.subplots(1,2,figsize=(WIDTH,3.4));fig.subplots_adjust(left=.075,right=.87,bottom=.255,top=.78,wspace=.4)
    confrows=[]
    for ai,(ax,v) in enumerate(zip(axes,['residual_fusion','resnet50'])):
        cms=np.array([metric[v,s,'overall','ALL']['confusion'] for s in seed_map[v]],dtype=float)
        cm=(100*cms/cms.sum(axis=2,keepdims=True)).mean(axis=0)
        im=ax.imshow(cm,cmap='Blues',vmin=0,vmax=100,interpolation='nearest')
        for i,true in enumerate(tx):
            for j,pred in enumerate(tx):
                ax.text(j,i,f'{cm[i,j]:.1f}',ha='center',va='center',fontsize=5.6,color='white' if cm[i,j]>=63 else '#243449')
                confrows.append(dict(variant=v,true_class=true,predicted_class=pred,mean_row_percentage=float(cm[i,j])))
        ax.set_xticks(range(6),tx,rotation=60,ha='right',fontsize=6);ax.set_yticks(range(6),tx,fontsize=6);ax.tick_params(length=0)
        ax.set_xlabel('Predicted identity',fontsize=7)
        if ai==0:ax.set_ylabel('True identity',fontsize=7)
        for spine in ax.spines.values():spine.set_visible(False)
        panel(ax,'ab'[ai],labels[v])
    cax=fig.add_axes([.90,.29,.016,.47]);cb=fig.colorbar(im,cax=cax);cb.set_ticks([0,50,100]);cb.set_label('Row percentage (%)',fontsize=7,labelpad=4)
    foot(fig,'Completed-seed mean row-normalized confusion; 28,000 packets per class and model.',.033)
    csvwrite(out/'source_data/confusion_mean_row_percentages.csv',confrows)
    save(fig,'fig06_confusion_matrices',
         'Mean row-normalized clean-test confusion matrices for CVS (a) and ResNet-50-1D (b). Actual n appears above each matrix. Rows are true identities and columns are predictions. Percentages are normalized per seed and then averaged; each class has 28,000 packets per model. These are descriptive model comparisons, not ensemble predictions.',
         ['confusion_mean_row_percentages.csv'])

    common=('All figures use the fixed ManySig clean benchmark from run '+raw['run_id']+'. '
        +((raw['snapshot_hkt']+' snapshot. '+n_context+'. Only fully independently scored seeds are included; each network has its own completion count. ') if progress_mode else 'Four model seeds share the same split. ')+
        'Every model uses the same 168,000 physical test packets. '
        'Source training uses receivers 1/3/4/6/8 and days 1/2/3, with 6,300 labelled source packets; '
        '56,700 unlabelled packets are unused and 27,000 source-validation packets are not trained on. '
        'All models are trained from scratch using only identity cross-entropy for a fixed 200 epochs (10,000 updates), '
        'then independently scored after predictions are frozen. Test scores do not select checkpoints or trigger reruns. '
        'Error bars show sample SD across model seeds (ddof=1), not standard error or confidence intervals. '
        'No hypothesis tests, p-values or significance stars are shown. Test packets are shared across models and seeds; '
        'model seeds are not independent dataset draws. All completed seeds, including collapsed results, are retained. '
        'This is a previously exposed proxy benchmark, with architecture/input/head adaptations and a common fixed optimization budget; '
        'it is not new blind testing, an architecture-specific best-tuned comparison or satellite-channel validation.')
    manifest=dict(run_id=raw['run_id'],status='VERIFIED',scope='clean only, fixed E200 completed-seed snapshot',
                  seeds_by_network={v:seed_map[v] for v in present},snapshot_hkt=raw.get('snapshot_hkt'),query_count_per_model=168000,figures=records,statistical_context=common,
                  export=dict(font='Arial',editable_svg_text=True,pdf_fonttype=42,png_dpi=600,
                              width_mm=183,backend='Python/matplotlib'),
                  validation=dict(score_csv_reconciled=True,all_seeds_retained=True,svg_text_checked=True,
                                  pdf_text_checked=True,pdf_width_checked=True,png_dpi_checked=True))
    (out/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    captions='# Figure legends\n\n'+common+'\n\n'
    for i,r in enumerate(records,1):captions+=f'## Figure {i}: {r["stem"]}\n\n{r["caption"]}\n\nSource data: '+', '.join(r['source_data'])+'.\n\n'
    (out/'captions_en.md').write_text(captions,encoding='utf-8')
    readme='''# 论文图表使用说明

本图包对应固定 E200、仅 clean 测试的已完成种子快照，实际种子数见各图标签和源数据，不能与 E20 实验混用。

1. fig01：总体 Accuracy/Macro-F1，保留每个种子和均值 ± 样本标准差。
2. fig02：7 个目标接收机的准确率热图，颜色固定为 0 至 100%。
3. fig03：参数量与准确率；参数轴为对数，编号对应完整网络名称。
4. fig04：1D/2D、复数/实数的同 seed 配对准确率差。
5. fig05：CVS 的接收机/发射机分层表现，保留全部已完成种子。
6. fig06：CVS 与 ResNet-50-1D 的平均行归一化混淆矩阵。

投稿优先使用 pdf/ 中的矢量文件；编辑标签、颜色和排版使用 svg/，其中字体保留为可编辑文本。png/ 是 600 dpi 位图；preview/ 是便于浏览的预览。所有图宽 183 mm，调整到单栏时应重新排版，避免直接缩小导致文字难读。实际投稿尺寸和字体要求按拟投期刊确认。

captions_en.md 提供英文图注及实验口径，source_data/ 提供汇总与逐种子数据，manifest.json 提供导出和验证记录。Transformer 的退化种子未剔除；均值误差线是 seed 间标准差，不是置信区间。本次未进行显著性检验。

在安装 NumPy、matplotlib、Pillow 和 pypdf 的 Python 环境中，可在图包目录执行：

```text
python plot_standard_ce_clean.py --results source_data --output ../new_figure_directory
```

source_data/verified_results.json 只保留绘图所需的评分与混淆矩阵，不包含远端进程、账号或运行路径。

这些图只支持既定预算、既定划分下的 clean 比较，不能作为真实在轨或星地信道鲁棒性的证据。2D 的 16×16 IQ reshape、复数层/分类头适配、统一训练日程等比较边界须在论文中披露。
'''
    (out/'README_zh.md').write_text(readme,encoding='utf-8')
    shutil.copyfile(Path(__file__),out/'plot_standard_ce_clean.py')
    # All visual composition, including the contact-sheet preview, uses matplotlib.
    fig,axes=plt.subplots(3,2,figsize=(12,13),layout='constrained')
    for ax,r in zip(axes.flat,records):
        with Image.open(out/'preview'/f'{r["stem"]}.png') as im:ax.imshow(np.asarray(im))
        ax.axis('off')
    fig.savefig(out/'preview/all_figures.png',dpi=160);plt.close(fig)
    merged=PdfWriter()
    for record in records:merged.append(out/'pdf'/f'{record["stem"]}.pdf')
    with (out/'figures_all.pdf').open('wb') as f:merged.write(f)
    assert len(PdfReader(out/'figures_all.pdf').pages)==6
    import zipfile
    archive=out/'standard_ce_clean_figure_bundle.zip'
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for path in sorted(out.rglob('*')):
            if path.is_file() and path!=archive:z.write(path,path.relative_to(out).as_posix())
    with zipfile.ZipFile(archive) as z:assert z.testzip() is None
    print(json.dumps(dict(status='VERIFIED',output=str(out),figures=len(records),formats=['PDF','SVG','PNG600dpi'],
                          source_data_files=len(list((out/'source_data').glob('*.csv')))),ensure_ascii=False))

if __name__=='__main__':main()
