"""Publication figures from complete PAMF-REPL and verified external Full tables."""
import argparse,csv,gc,json,statistics
from pathlib import Path
ROOT=Path('E:/type10-7')
DEFAULT=ROOT/'local_artifacts/pamf_repl_completion_20261024/final_analysis_20261025'
OUT=ROOT/'exports/unified_figures/20261025-PAMF-REPL-complete-paper-r01'
TASKS=('cross_date','cross_rx','cross_both')
METHODS=('riei','drift','ibrff','sdg','PAMF-supervised','PAMF-REPL')
LABELS=('RIEI Full + PL','DRIFT-v2 Full + PL','IB-RFF Full + PL','SDG Full + PL','PAMF-Sup','PAMF-REPL')
COLORS=('#52769C','#8B6B99','#C18D53','#639A92','#A6B9CC','#254C78')

def read(p):
    with Path(p).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def write(p,rows):
    with Path(p).open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def build(source,output,raster):
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    from PIL import Image
    from matplotlib import font_manager
    source,output=Path(source),Path(output)
    for sub in ('pdf','svg','preview','png','tiff','tables','source_data','reproduce','IEEE_experiments'): (output/sub).mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'Times New Roman','font.size':8,'axes.labelsize':8,'xtick.labelsize':7,'ytick.labelsize':7,'legend.fontsize':6.5,'svg.fonttype':'none','pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.6,'savefig.facecolor':'white'})
    assert 'times' in font_manager.findfont('Times New Roman').lower()
    main=read(source/'paper_all_recipes.csv');lookup={(r['protocol'],r['method'],r['cell'],r['test'],r['readout'],r['view'],r['metric']):r for r in main}
    own=read(source/'summary_overall.csv');ownmap={(r['protocol'],r['method'],r['test'],r['readout'],r['view'],r['metric']):r for r in own}
    effects=read(source/'paired_effects.csv');core=read(source/'core_effects.csv');timing=read(source/'timing_recomputed.csv');figures=[]
    def get(p,method,t='cross_rx',rd=None,v='six_practical_mean',metric='accuracy'):
        cell='native' if method.startswith('PAMF-') else 'D';rd=rd or ('DURA-L' if method=='PAMF-supervised' else 'DURA-LU')
        return lookup[p,method,cell,t,rd,v,metric]
    def avg(rows):return [float(r['mean']) for r in rows]
    def sd(rows):return [float(r['sd']) for r in rows]
    def save(fig,slug,claim,caption,data,role='main'):
        for old,new in {'Full111 minus011':'Full 111 minus 011','strict384':'strict 384','Identical720':'Identical 720',',90 ':', 90 ', ',11 ':', 11 ','all11':'all 11'}.items():caption=caption.replace(old,new)
        stem='Fig%02d_'%(len(figures)+1)+slug
        # Remove locator ticks outside the actual view before extent inspection;
        # Matplotlib otherwise retains invisible off-axis Text objects.
        for ax in fig.axes:
            lo,hi=ax.get_xlim();ax.set_xticks([v for v in ax.get_xticks() if lo<=v<=hi])
            lo,hi=ax.get_ylim();a,b=sorted((lo,hi));ax.set_yticks([v for v in ax.get_yticks() if a<=v<=b])
        fig.canvas.draw();renderer=fig.canvas.get_renderer();box=fig.bbox
        for artist in fig.findobj(matplotlib.text.Text):
            if artist.get_visible() and artist.get_text():
                b=artist.get_window_extent(renderer)
                if not(b.x0>=-2 and b.y0>=-2 and b.x1<=box.width+2 and b.y1<=box.height+2):raise ValueError((stem,artist.get_text(),str(b)))
        fig.savefig(output/'pdf'/(stem+'.pdf'));fig.savefig(output/'svg'/(stem+'.svg'));fig.savefig(output/'preview'/(stem+'.png'),dpi=180)
        pixels=None
        if raster:
            fig.savefig(output/'png'/(stem+'.png'),dpi=4800);Image.MAX_IMAGE_PIXELS=None
            with Image.open(output/'png'/(stem+'.png')) as image:
                pixels=list(image.size);image.save(output/'tiff'/(stem+'.tiff'),compression='tiff_lzw',dpi=(4800,4800))
        write(output/'tables'/(stem+'.csv'),data)
        figures.append(dict(stem=stem,claim=claim,caption_en=caption,role=role,font='Times New Roman',dpi=4800 if raster else 180,pixels=pixels,one_chart=True,no_inside_title=True,no_footnotes=True))
        plt.close(fig);gc.collect();print(stem,flush=True)
    common='Three F seeds, mean and sample SD; S6 is averaged within seed. G is fixed by contract/role unless stated. SD is not a confidence interval or significance test. '
    for task in TASKS:
        for restored in (False,True):
            rows=[get('F0',method,task,None if restored else 'raw') for method in METHODS]
            fig,ax=plt.subplots(figsize=(3.5,2.8));fig.subplots_adjust(left=.37,right=.96,bottom=.17,top=.97)
            ax.barh(range(6),avg(rows),xerr=sd(rows),color=COLORS,height=.6,error_kw={'lw':.65,'capsize':1.5});ax.set(yticks=range(6),yticklabels=LABELS,xlim=(0,100),xlabel='Six-channel mean accuracy (%)');ax.invert_yaxis();ax.grid(axis='x',alpha=.15);ax.set_axisbelow(True)
            save(fig,'F0_'+task+('_restored' if restored else '_raw'),'Compare fixed complete recipes on '+task,common+'F0 '+task+'. External cell D is fixed, not target-selected. Sup uses G-L; external PL/REPL use G-LU. Complete-recipe differences are not single-component causal effects.',rows)
    for slug,pids,x,xlabels in [('source', ['m%d-natural'%n for n in (1,2,3,4,6)],[1,2,3,4,6],['1','2','3','4','6']),('labels',['label01','label05','F0'],[1,5,10],['1','5','10']),('receivers',['F0','F1_rx_repartition_r01','F2_rx_repartition_r01'],[0,1,2],['F0','F1','F2'])]:
        fig,ax=plt.subplots(figsize=(3.5,2.8));fig.subplots_adjust(left=.16,right=.97,bottom=.2,top=.76);data=[]
        for method,label,color in zip(METHODS,LABELS,COLORS):
            rows=[get(p,method) for p in pids];data.extend(rows);ax.errorbar(x,avg(rows),yerr=sd(rows),color=color,marker='o',markersize=3,lw=.9,capsize=1.4,label=label)
        ax.set(xticks=x,xticklabels=xlabels,ylabel='S6 accuracy (%)',xlabel={'source':'Number of source receivers','labels':'Labeled fraction of training pool (%)','receivers':'Source-receiver partition'}[slug]);ax.grid(alpha=.15);ax.legend(loc='lower center',bbox_to_anchor=(.5,1.01),ncol=2,frameon=False,columnspacing=.7,handlelength=1.2)
        save(fig,slug+'_cross_rx_restored','Measure robustness without selecting favorable source sets',common+'Cross-RX endpoint, matching restoration. Source curve uses natural budgets and fixed six target RX, m=1,2,3,4,6 only; F0 is not a five-source point. Label and RX partitions retain their registered physical roles. All nonmonotonic values remain.',data)
    rows=[get('ManyRx',m,metric='macro_f1') for m in METHODS]
    fig,ax=plt.subplots(figsize=(3.5,2.8));fig.subplots_adjust(left=.37,right=.96,bottom=.17,top=.97);ax.barh(range(6),avg(rows),xerr=sd(rows),color=COLORS,height=.6,error_kw={'lw':.65,'capsize':1.5});ax.set(yticks=range(6),yticklabels=LABELS,xlim=(0,100),xlabel='ManyRx S6 Macro-F1 (%)');ax.invert_yaxis();ax.grid(axis='x',alpha=.15);ax.set_axisbelow(True)
    save(fig,'ManyRx_macroF1','Retain the full ten-class ManyRx evaluation',common+'ManyRx cross-RX Macro-F1 uses all ten registered TX, with unequal cell sizes; no pooling with six-class ManySig.',rows)
    scenes=('clean','practical_high','practical_mid','practical_low_suburban','practical_high_urban','practical_mid_urban','practical_low_urban');short=('Clean','High S','Mid S','Low S','High U','Mid U','Low U')
    fig,ax=plt.subplots(figsize=(3.5,2.9));fig.subplots_adjust(left=.16,right=.97,bottom=.21,top=.76);data=[]
    for method,label,color in zip(METHODS,LABELS,COLORS):
        rows=[get('F0',method,v=v) for v in scenes];data.extend(rows);ax.errorbar(range(7),avg(rows),yerr=sd(rows),color=color,marker='o',markersize=3,lw=.9,capsize=1,label=label)
    ax.set(xticks=range(7),xticklabels=short,ylabel='Accuracy (%)');ax.tick_params(axis='x',rotation=35);ax.grid(alpha=.15);ax.legend(loc='lower center',bbox_to_anchor=(.5,1.01),ncol=2,frameon=False,columnspacing=.7,handlelength=1.2)
    save(fig,'F0_seven_views_restored','Show channel-specific results rather than only their mean',common+'F0 cross-RX; practical channel views are simulations. Clean is also restored; there is no scene-aware bypass.',data)
    fig,ax=plt.subplots(figsize=(3.5,2.6));fig.subplots_adjust(left=.16,right=.97,bottom=.24,top=.77);data=[]
    for off,(rd,v,label,color) in zip((-.2,0,.2),(('raw','clean','Raw clean','#8B6B99'),('raw','six_practical_mean','Raw S6','#52769C'),('DURA-LU','six_practical_mean','G-LU S6','#639A92'))):
        rows=[next(r for r in core if r['effect']==name and r['test']=='cross_rx' and r['readout']==rd and r['view']==v and r['metric']=='accuracy') for name in ('full_minus_no_L_CRIC','full_minus_no_L_consistency','full_minus_no_U')]
        data.extend(rows);ax.errorbar(np.arange(3)+off,avg(rows),yerr=sd(rows),fmt='o',ms=3,capsize=2,color=color,label=label)
    ax.axhline(0,color='.6',lw=.6);ax.set(xticks=range(3),xticklabels=['A: L CRIC','B: L consistency','C: REPL U'],ylabel='Full minus removal (pp)');ax.legend(loc='lower center',bbox_to_anchor=(.5,1.02),ncol=3,frameon=False,handlelength=1)
    save(fig,'core_conditional_effects','Distinguish conditional contributions from marginal effects',common+'Full111 minus011/101/110. A is supervised CRIC, B supervised paired KL, C the entire REPL U objective. Full conditional effects may differ from eight-cell marginal effects. C0 opens no U loader.',data)
    fig,ax=plt.subplots(figsize=(3.5,2.6));fig.subplots_adjust(left=.17,right=.97,bottom=.21,top=.79);data=[];bits=['000','001','010','011','100','101','110','111']
    for rd,color in zip(('raw','DURA-L','DURA-LU'),('#8B6B99','#52769C','#639A92')):
        rows=[ownmap['F0',('REPL' if b=='111' else 'supervised' if b=='110' else 'ABC'+b),'cross_rx',rd,'six_practical_mean','accuracy'] for b in bits];data.extend(rows);ax.errorbar(range(8),avg(rows),yerr=sd(rows),marker='o',markersize=3,capsize=1,lw=.8,color=color,label=rd)
    ax.set(xticks=range(8),xticklabels=bits,xlabel='A / B / C functional groups',ylabel='S6 accuracy (%)');ax.grid(alpha=.15);ax.legend(loc='lower center',bbox_to_anchor=(.5,1.02),ncol=3,frameon=False)
    save(fig,'core_eight_cells','Expose interactions and all eight functional combinations',common+'F0 cross-RX; 000 is a matched 768-L-forward multi-view CE skeleton, not strict384-forward CE. A fixed G-LU still uses U when F has C0.',data,'appendix')
    fig,ax=plt.subplots(figsize=(3.5,2.8));fig.subplots_adjust(left=.29,right=.97,bottom=.18,top=.8);data=[]
    for off,(rd,v,label,color) in zip((-.12,.12),(('raw','clean','Raw clean','#8B6B99'),('DURA-LU','six_practical_mean','G-LU S6','#52769C'))):
        rows=[next(r for r in effects if r['protocol']=='F0' and r['effect']=='REPL_minus_without_'+b and r['test']=='cross_rx' and r['readout']==rd and r['view']==v and r['metric']=='accuracy') for b in ('style','guard','joint','extrapolation')];data.extend(rows);ax.errorbar(avg(rows),np.arange(4)+off,xerr=sd(rows),fmt='o',ms=3,capsize=1.5,color=color,label=label)
    ax.axvline(0,color='.6',lw=.6);ax.set(yticks=range(4),yticklabels=['U style','Gradient guard','Joint contrast','RX extrapolation'],xlabel='Full minus removal (pp)');ax.invert_yaxis();ax.legend(loc='lower center',bbox_to_anchor=(.5,1.01),ncol=2,frameon=False)
    save(fig,'REPL_inner_mechanisms','Test the actual four components of the selected REPL recipe',common+'F0 cross-RX, fixed E50 and default head. No-extrapolation reuses the exact matching historical Joint160 F; no native EMA/gate ablations are relabeled as REPL.',data)
    arms=('full','SIG_TIME','NO_TIME','SHUFFLE_TIME','NO_SIGNATURE','NO_PA','NO_REFERENCE','WRONG_REFERENCE','COV_X3');arm_labels=('Full','Single route','No time','Shuffle time','No signature','No PA','No reference','Wrong reference','COV X3')
    for prefix,name in [('REPL','REPL'),('CE','Strict CE')]:
        fig,ax=plt.subplots(figsize=(3.5,2.9));fig.subplots_adjust(left=.32,right=.97,bottom=.16,top=.81);data=[]
        for off,(rd,label,color) in zip((-.12,.12),(('raw','Raw','#8B6B99'),('DURA-LU' if prefix=='REPL' else 'DURA-L','Matching G','#52769C'))):
            methods=[('REPL' if a=='full' else 'REPL_'+a) if prefix=='REPL' else 'CE_'+a for a in arms]
            rows=[ownmap['F0',m,'cross_rx',rd,'six_practical_mean','accuracy'] for m in methods];data.extend(rows);ax.errorbar(avg(rows),np.arange(9)+off,xerr=sd(rows),fmt='o',ms=2.5,capsize=1,color=color,label=label)
        ax.set(yticks=range(9),yticklabels=arm_labels,xlabel='Six-channel mean accuracy (%)');ax.invert_yaxis();ax.grid(axis='x',alpha=.15);ax.legend(loc='lower center',bbox_to_anchor=(.5,1.01),ncol=2,frameon=False)
        save(fig,prefix+'_structure','Evaluate architecture under '+name+' with matching restoration',common+name+' structure layer. REPL has768 L packet forwards/update and U; strict CE has384 and no U/EMA/CRIC/KL. Compare variants within layer, not as an exposure-matched cross-layer causal contrast.',data)
    gmethods=['REPL']+sorted({r['method'] for r in own if r['group']=='G_component'});glabels=['DURA full']+[m.removeprefix('G_').replace('_',' ') for m in gmethods[1:]];rows=[]
    for m in gmethods:
        rd='DURA-LU' if m=='REPL' else m.removeprefix('G_');rows.append(ownmap['F0',m,'cross_rx',rd,'six_practical_mean','accuracy'])
    fig,ax=plt.subplots(figsize=(3.5,2.8));fig.subplots_adjust(left=.38,right=.97,bottom=.17,top=.97);ax.errorbar(avg(rows),range(8),xerr=sd(rows),fmt='o',ms=3,capsize=1.5,color='#52769C');ax.set(yticks=range(8),yticklabels=glabels,xlabel='S6 accuracy (%)');ax.invert_yaxis();ax.grid(axis='x',alpha=.15)
    save(fig,'existing_G_with_REPL','Reconnect existing restorer controls to the selected classifier',common+'Existing G weights only; no G retraining. Default uses fixed G1 across F seeds, whereas seven variants use their paired G seeds. This is not pure G variance or a fully replicated copy-by-KD factorial.',rows)
    cross=[r for r in read(source/'m3_crossed_cells.csv') if r['test']=='cross_rx' and r['view']=='six_practical_mean' and r['metric']=='accuracy'];arr=np.array([[float(next(r['percent'] for r in cross if r['F_seed']==str(f) and r['G_seed']==str(g))) for g in (2026092701,2026092702,2026092703)] for f in (2026092701,2026092702,2026092703)])
    fig,ax=plt.subplots(figsize=(3.2,2.7));fig.subplots_adjust(left=.22,right=.92,bottom=.2,top=.95);ax.imshow(arr,cmap='Blues',vmin=float(arr.min())-.2,vmax=float(arr.max())+.2)
    for i in range(3):
        for j in range(3):ax.text(j,i,f'{arr[i,j]:.2f}',ha='center',va='center',color='white' if arr[i,j]>np.mean(arr) else '#18334C')
    ax.set(xticks=range(3),yticks=range(3),xticklabels=['01','02','03'],yticklabels=['01','02','03'],xlabel='G training seed suffix',ylabel='F training seed suffix')
    save(fig,'m3_crossed_seed','Show the complete three-F by three-G sensitivity grid','m3-natural cross-RX S6. Each cell is one frozen F/G pair, without a within-cell SD. Three diagonal pairs are three complete-system seeds; nine cells are not nine independent replicas. Numerals are authoritative.',cross,'appendix')
    fig,ax=plt.subplots(figsize=(3.5,2.7));fig.subplots_adjust(left=.18,right=.97,bottom=.19,top=.77);data=[]
    for method,label,color in [('supervised','Sup','#A6B9CC'),('ordinary_PL','PL','#8B6B99'),('REPL','REPL','#254C78')]:
        for suffix,style in [('F','--'),('GF','-')]:
            panel='REPL_structure' if method=='REPL' else 'main_controls';rows=sorted([r for r in timing if r['panel']==panel and r['case']==method+'_'+suffix],key=lambda r:int(r['batch']));data.extend(rows);ax.plot([int(r['batch']) for r in rows],[float(r['throughput_IQ_s']) for r in rows],color=color,ls=style,marker='o',ms=3,lw=.9,label=label+' '+suffix)
    ax.set(xscale='log',yscale='log',xticks=[1,8,32,64,128],xticklabels=['1','8','32','64','128'],xlabel='Batch size',ylabel='Throughput (IQ/s)');ax.grid(alpha=.15);ax.legend(loc='lower center',bbox_to_anchor=(.5,1.01),ncol=3,frameon=False)
    save(fig,'measured_throughput','Measure operational throughput of the updated classifier and G/F execution','Exclusive RTX3090, FP32/TF32off, five rounds per batch. F is eager and GF uses the established optimized graph engine; their difference is an execution-path benchmark, not an isolated algorithmic speedup from adding G. Cached classification is excluded.',data)
    for prefix,panel in [('REPL','REPL_structure'),('CE','strict_structure')]:
        fig,ax=plt.subplots(figsize=(3.5,2.9));fig.subplots_adjust(left=.32,right=.97,bottom=.17,top=.81);data=[]
        methods=[('REPL' if a=='full' else 'REPL_'+a) if prefix=='REPL' else 'CE_'+a for a in arms]
        for off,suffix,color in [(-.12,'F','#8B6B99'),(.12,'GF','#52769C')]:
            rows=[next(r for r in timing if r['panel']==panel and r['case']==m+'_'+suffix and r['batch']=='1') for m in methods];data.extend(rows);ax.errorbar([float(r['wall_median_ms']) for r in rows],np.arange(9)+off,xerr=[float(r['round_median_sd']) for r in rows],fmt='o',ms=2.5,capsize=1,color=color,label=suffix)
        ax.set(yticks=range(9),yticklabels=arm_labels,xlabel='B1 wall latency (ms)');ax.invert_yaxis();ax.grid(axis='x',alpha=.15);ax.legend(loc='lower center',bbox_to_anchor=(.5,1.01),ncol=2,frameon=False)
        save(fig,prefix+'_structure_latency','Report architecture latency with its measured execution path','Five rounds are timing repetitions, not model seeds. Points are pooled wall medians and bars are SD of round medians. F eager and GF optimized are disclosed separately. Co-resident baseline/incremental memory are retained; no isolated-model VRAM claim.',data,'appendix')
    physical=read(source/'source_interventions.csv');interventions=list(dict.fromkeys(r['intervention'] for r in physical));short=['Identity','Gain .5','Gain 2','Phase pi/2','CFO -100k','CFO -25k','CFO +25k','CFO +100k','FIR echo','PA .05','PA .1'];fig,ax=plt.subplots(figsize=(3.5,2.9));fig.subplots_adjust(left=.17,right=.97,bottom=.3,top=.8);data=[]
    for method,label,color in [('supervised','Sup','#A6B9CC'),('ordinary_PL','PL','#8B6B99'),('REPL','REPL','#254C78')]:
        vals=[];errors=[]
        for intervention in interventions:
            rr=[r for r in physical if r['method']==method and r['intervention']==intervention];assert len(rr)==3;data.extend(rr);a=[100*float(r['default_head_flip_fraction']) for r in rr];vals.append(statistics.mean(a));errors.append(statistics.stdev(a))
        ax.errorbar(range(11),vals,yerr=errors,color=color,marker='o',ms=2.5,lw=.8,capsize=1,label=label)
    ax.set(xticks=range(11),xticklabels=short,ylabel='Default-head flip fraction (%)');ax.tick_params(axis='x',rotation=60);ax.legend(loc='lower center',bbox_to_anchor=(.5,1.01),ncol=3,frameon=False);ax.grid(alpha=.15)
    save(fig,'source_physical_sensitivity','Bound physical invariance claims with source-only interventions','Identical720 labeled-source packets,90 TX/RX/day strata,11 interventions, three F seeds. No U/target or fitted probe. A synthetic PA perturbation is not assumed identity-preserving; this is sensitivity evidence, not complete hardware disentanglement.',data,'appendix')
    pids=['F0','F1_rx_repartition_r01','F2_rx_repartition_r01','ManyRx','label01','label05','m1-natural','m2-natural','m3-natural','m4-natural','m6-natural']
    fig,ax=plt.subplots(figsize=(3.5,2.7));fig.subplots_adjust(left=.17,right=.97,bottom=.23,top=.79);data=[]
    for rd,color in [('DURA-L','#8B6B99'),('DURA-LU','#52769C')]:
        rows=[next(r for r in effects if r['protocol']==p and r['effect']=='PL_minus_Sup' and r['test']=='cross_rx' and r['readout']==rd and r['view']=='six_practical_mean' and r['metric']=='accuracy') for p in pids];data.extend(rows);ax.errorbar(range(11),avg(rows),yerr=sd(rows),color=color,marker='o',ms=2.5,lw=.8,capsize=1,label=rd)
    ax.axhline(0,color='.6',lw=.6);ax.set(xticks=range(11),xticklabels=['F0','F1','F2','MR','L1','L5','m1','m2','m3','m4','m6'],ylabel='Ordinary PL minus Sup (pp)');ax.tick_params(axis='x',rotation=40);ax.legend(loc='lower center',bbox_to_anchor=(.5,1.01),ncol=2,frameon=False);ax.grid(alpha=.15)
    save(fig,'PAMF_fairness_fixed_G','Separate F pseudo-labeling from G unlabeled-IQ permissions',common+'New matched PAMF Sup/ordinary-PL F evaluated with both G-L and G-LU on all11 contracts. PL F still uses U with G-L; only Sup F plus G-L is a completely L-only system. Negative effects remain.',data,'appendix')
    for family,label,color in zip(METHODS[:4],LABELS[:4],COLORS[:4]):
        fig,ax=plt.subplots(figsize=(3.5,2.6));fig.subplots_adjust(left=.17,right=.97,bottom=.24,top=.8);data=[]
        for restored,style in [(False,'--'),(True,'-')]:
            rows=[lookup['F0',family,c,'cross_rx',('DURA-LU' if c in ('B','D') else 'DURA-L') if restored else 'raw','six_practical_mean','accuracy'] for c in ('A','B','C','D')];data.extend(rows);ax.errorbar(range(4),avg(rows),yerr=sd(rows),color=color if restored else '.5',ls=style,marker='o',ms=3,capsize=1.5,label='Matching G' if restored else 'Raw')
        ax.set(xticks=range(4),xticklabels=['Off/Off','Off/PL','On/Off','On/PL'],xlabel='Additional channel / ordinary PL',ylabel='S6 accuracy (%)');ax.grid(alpha=.15);ax.legend(loc='lower center',bbox_to_anchor=(.5,1.01),ncol=2,frameon=False)
        save(fig,family+'_Full_four_cells','Retain the complete external augmentation/PL factorial',common+label+' on F0 cross-RX. All four cells are Full, not CE-only. Existing verified external aggregates are reused; A/C restoration uses G-L, B/D G-LU, so restored cross-PL contrasts also change G permissions.',data,'appendix')
    (output/'figure_manifest.json').write_text(json.dumps(dict(method='PAMF-REPL',figures=figures,statistics='three actual F seeds; sample SD; within-seed scene average',historical_T1_relabelled=False),indent=2)+'\n')
    (output/'figure_captions_en.md').write_text('\n\n'.join('**'+r['stem']+'** '+r['caption_en'] for r in figures)+'\n',encoding='utf-8')
    write(output/'tables/paper_main_recipes.csv',[r for r in main if (r['method'] in METHODS and r['cell'] in ('D','native')) and r['view'] in ('clean','six_practical_mean')])
    print(json.dumps(dict(status='FIGURES_WRITTEN',figures=len(figures),output=str(output))),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,default=DEFAULT);p.add_argument('--output',type=Path,default=OUT);p.add_argument('--raster',action='store_true');a=p.parse_args();build(a.source,a.output,a.raster)
