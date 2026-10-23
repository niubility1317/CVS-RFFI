"""Standalone publication figures from the completed PAMF snapshot.

All drawing, previews, vector exports and high-resolution rasterization use Python.
"""
from __future__ import annotations
import argparse
import csv
import json
import math
from pathlib import Path
import shutil
import statistics as stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Patch
from pypdf import PdfWriter
import numpy as np
from collect_completed_pamf import PROTOCOLS, SEEDS, SIX, VIEWS, read, write, save_csv
from analyze_completed_pamf import one, rows
from paper_publication_style import apply_style, canvas, COLORS, CMAP, INK, MUTED, grid, pale, contrast
from plot_pamf_paper_figures import export, ranked_bars

SCENES=('Clean','Suburban\nhigh','Suburban\nmid','Suburban\nlow','Urban\nhigh','Urban\nmid','Urban\nlow')
REGIMES=('native_cric','matched_sat_supervised','matched_sat_plain_pl')
REGIME_NAMES=('T1','Supervised','Ordinary PL')
REGIME_COLORS=(COLORS['pamf'],COLORS['riei'],COLORS['sdg'])
READOUTS=('raw','DURA-L','DURA-LU')
ARCHS=('X3_CE','SIG_TIME','NO_TIME','SHUFFLE_TIME','NO_SIGNATURE','NO_PA','NO_REFERENCE','WRONG_REFERENCE','COV_X3')
ARCH_NAMES={'X3_CE':'Full X3-CE','SIG_TIME':'Single-view signature','NO_TIME':'Without time coordinates',
    'SHUFFLE_TIME':'Shuffled time coordinates','NO_SIGNATURE':'Without signature branch','NO_PA':'Without PA branch',
    'NO_REFERENCE':'Without reference branch','WRONG_REFERENCE':'Perturbed reference','COV_X3':'Covariance substitute'}
FINE_NAMES=('Shallow CRIC','Intermediate CRIC','Final-layer CRIC','Without paired KL','Single stochastic forward',
    'Single signature view','Without time coordinates','Without signature','Without PA','Without reference',
    'Student PL source','Without adjacent gate','Without strong-view gate','Without U entropy','CE recipe',
    'Without CRIC','Raw replaces As views','As KL restored','Reduced raw CE dose','Supervised T1','Ordinary PL')
G_NAMES={'raw':'Raw T1','default':'DURA','coarse_only':'Coarse only','prior_no_CG':'Without CG',
    'no_image_action':'Without image action','no_innovation_input':'Without residual input',
    'copy_no_KD':'Copy without KD','random_with_KD':'Random + KD','random_no_KD':'Random without KD'}
FAMILIES=('riei','drift','ibrff','sdg')
FAM_NAMES={'riei':'RIEI','drift':'DRIFT','ibrff':'IB-RFF','sdg':'SDG'}
INTERVENTIONS=('identity','flat_gain_0.5','flat_gain_2.0','flat_phase_pi2','relative_CFO_-100000',
    'relative_CFO_-25000','relative_CFO_25000','relative_CFO_100000','FIR_delay3_echo02',
    'synthetic_PA_alpha_0.05','synthetic_PA_alpha_0.1')
INT_NAMES=('Identity','Gain\n0.5','Gain\n2.0','Phase\npi/2','CFO\n-100k','CFO\n-25k','CFO\n+25k',
    'CFO\n+100k','FIR\necho','PA\n0.05','PA\n0.1')
COMMON=dict(test='main',dimension='overall',stratum='ALL',metric='accuracy')
FIXED_G='Three F model seeds conditional on one fixed G per data contract. Whiskers are sample SD (ddof=1), not confidence intervals or a significance test.'


def forest(labels,records,xlabel,limits=(0,100),highlight=(),height=None,colors=None):
    n=len(labels);height=height or max(83,min(145,34+5.5*n))
    fig,ax=canvas('',height,(.33,.16,.47,.70))
    low=min(float(r['mean'])-float(r['sd']) for r in records)
    high=max(float(r['mean'])+float(r['sd']) for r in records)
    lo=min(limits[0],math.floor(low));hi=max(limits[1],math.ceil(high))
    ax.set(yticks=range(n),yticklabels=labels,ylim=(n-.45,-.6),xlim=(lo,hi),xlabel=xlabel)
    ax.tick_params(axis='y',length=0,pad=7);ax.spines['left'].set_visible(False);grid(ax,'x')
    if lo<0<hi:ax.axvline(0,color=MUTED,lw=.75,ls=(0,(3,3)))
    for i,r in enumerate(records):
        c=colors[i] if colors else COLORS['pamf'] if i in highlight else COLORS['riei']
        if i in highlight:ax.axhspan(i-.40,i+.40,color=pale(c,.065))
        m,sd=float(r['mean']),float(r['sd'])
        ax.errorbar(m,i,xerr=sd,fmt='D',mfc=c if i in highlight else 'white',color=c,ms=4.5,lw=.9,capsize=2)
        ax.text(1.06,i,f'{m:+.2f} ± {sd:.2f}' if lo<0 else f'{m:.2f} ± {sd:.2f}',
            transform=ax.get_yaxis_transform(),va='center',fontsize=8.1,color=c)
        ax.get_yticklabels()[i].set_color(c)
    return fig


def matrix(values,labels,columns,unit='Accuracy (%)',signed=False,sds=None,height=125,limits=None):
    a=np.asarray(values,float);n,m=a.shape
    fig,ax=canvas('',height,(.25,.23,.70,.64))
    cmap=plt.get_cmap('RdBu_r').copy() if signed else CMAP.copy();cmap.set_bad('#F4F4F4')
    if signed:
        extent=max(abs(np.nanmin(a)),abs(np.nanmax(a)),.25) if limits is None else limits[1]
        step=10**math.floor(math.log10(extent))
        extent=math.ceil(extent/step*2)/2*step
        norm=TwoSlopeNorm(vmin=-extent,vcenter=0,vmax=extent)
    else:
        norm=matplotlib.colors.Normalize(*(limits or (0,100)))
    im=ax.pcolormesh(np.arange(m+1)-.5,np.arange(n+1)-.5,np.ma.masked_invalid(a),
        cmap=cmap,norm=norm,shading='flat',edgecolors='white',linewidth=.7,rasterized=False)
    ax.set(xlim=(-.5,m-.5),ylim=(n-.5,-.5),xticks=range(m),xticklabels=columns,yticks=range(n),yticklabels=labels)
    ax.tick_params(length=0,pad=7)
    for spine in ax.spines.values():spine.set_visible(False)
    for i in range(n):
        for j in range(m):
            value=a[i,j]
            text='N/A' if not math.isfinite(value) else f'{value:+.2f}' if signed else f'{value:.1f}'
            if sds is not None and math.isfinite(value):text+=f'\n± {sds[i][j]:.2f}'
            ax.text(j,i,text,ha='center',va='center',fontsize=7.8 if m>=10 or sds is not None else 8.1,
                linespacing=1.2,color=MUTED if not math.isfinite(value) else contrast(cmap(norm(value))))
    cax=fig.add_axes((.36,.10 if height<110 else .075,.52,.016),label='colorbar')
    cb=fig.colorbar(im,cax=cax,orientation='horizontal',
        ticks=np.linspace(norm.vmin,norm.vmax,5),format='%g');cb.outline.set_visible(False)
    cb.solids.set_rasterized(False);cb.set_label(unit,labelpad=3,fontsize=8);cb.ax.tick_params(labelsize=7.6,length=2,pad=2)
    return fig


def grouped_scene(t):
    fig,ax=canvas('',108,(.105,.22,.85,.66));x=np.arange(7)
    conditions=[('Raw T1','raw',-.23,COLORS['baseline']),('T1 + G-L','DURA-L',0,COLORS['riei']),
                ('PAMF','DURA-LU',.23,COLORS['pamf'])]
    for label,readout,offset,c in conditions:
        records=[one(t,'core',bits='111',readout=readout,view=v,**COMMON) for v in VIEWS]
        means=[r['mean'] for r in records];sds=[r['sd'] for r in records]
        ax.bar(x+offset,means,width=.21,color=c,edgecolor=c,linewidth=.4,yerr=sds,
            error_kw=dict(ecolor=INK,elinewidth=.7,capsize=1.7),label=label)
    ax.set(ylim=(0,100),yticks=range(0,101,20),xticks=x,xticklabels=SCENES,ylabel='Accuracy (%)')
    grid(ax);ax.tick_params(axis='x',length=0,pad=7)
    fig.legend(*ax.get_legend_handles_labels(),loc='center',bbox_to_anchor=(.52,.07),ncol=3)
    return fig


def source_curve(t,budget,readout):
    fig,ax=canvas('',105,(.105,.20,.85,.68));counts=[1,2,3,4,6]
    for regime,label,c,marker in zip(REGIMES,REGIME_NAMES,REGIME_COLORS,('D','o','s')):
        data=[one(t,'main',protocol=f'm{m}-{budget}',regime=regime,readout=readout,view=SIX,**COMMON) for m in counts]
        ax.errorbar(counts,[r['mean'] for r in data],yerr=[r['sd'] for r in data],label=label,color=c,
            marker=marker,ms=4.5,lw=1.2,capsize=2,elinewidth=.7,mfc=c if regime=='native_cric' else 'white')
    ax.set(xlim=(.7,6.3),ylim=(40,82),xticks=counts,yticks=range(40,81,10),xlabel='Number of source receivers',
        ylabel='Mean channel accuracy (%)');grid(ax,'both')
    fig.legend(*ax.get_legend_handles_labels(),loc='center',bbox_to_anchor=(.53,.075),ncol=3)
    return fig


def external_f0(t,readout,strict=False):
    fig,ax=canvas('',108,(.105,.22,.85,.66));positions=[0,1,2,3,4.2]
    for family,x in zip(FAMILIES,positions):
        for regime,dx in ([('supervised',0)] if strict else [('supervised',-.17),('pl',.17)]):
            r=one(t,'external_full',protocol='F0',family=family,regime=regime,variant='full',readout=readout,
                test='main',view=SIX,metric='accuracy')
            c=COLORS[family]
            ax.bar(x+dx,r['mean'],width=.29,color=pale(c) if regime=='supervised' and not strict else c,
                edgecolor=c,hatch='//' if regime=='supervised' and not strict else None,yerr=r['sd'],linewidth=.7,
                error_kw=dict(ecolor=INK,elinewidth=.7,capsize=2))
            ax.text(x+dx,r['mean']+r['sd']+1.4,f'{r["mean"]:.1f}',ha='center',fontsize=7.8,color=c)
    if strict:
        r=one(t,'gfactor',protocol_id='F0',cell='Sup+G-L',test='main',view=SIX,metric='accuracy')
    else:r=one(t,'main',protocol='F0',regime='native_cric',readout=readout,view=SIX,**COMMON)
    ax.bar(4.2,r['mean'],width=.29,color=COLORS['pamf'],yerr=r['sd'],
        error_kw=dict(ecolor=INK,elinewidth=.7,capsize=2))
    ax.text(4.2,r['mean']+r['sd']+1.4,f'{r["mean"]:.1f}',ha='center',fontsize=8,color=COLORS['pamf'])
    ax.set(xlim=(-.55,4.75),ylim=(0,100),yticks=range(0,101,20),xticks=positions,
        xticklabels=[FAM_NAMES[k] for k in FAMILIES]+['PAMF (Sup)' if strict else 'PAMF'],ylabel='Mean channel accuracy (%)')
    grid(ax);ax.tick_params(axis='x',length=0,pad=8)
    if not strict:
        fig.legend(handles=[Patch(fc=pale(MUTED),ec=MUTED,hatch='//',label='Full'),
            Patch(fc=MUTED,label='Full + PL'),Patch(fc=COLORS['pamf'],label='PAMF')],
            loc='center',bbox_to_anchor=(.53,.07),ncol=3)
    return fig


def render(source,out,preview_only,dpi):
    data=read(source);t=data['tables'];records=[];resources=[]
    analysis=read(out/'analysis/conclusions_data.json')
    ext_effects=[]
    with (out/'analysis/external_full_paired_effects.csv').open(encoding='utf-8-sig') as stream:
        for r in csv.DictReader(stream):
            for k in ('mean','sd'):r[k]=float(r[k])
            ext_effects.append(r)
    for folder in ('pdf','svg','png','tiff','preview'): (out/folder).mkdir(exist_ok=True)
    def save(fig,slug,title,caption,tables):
        records.append(export(fig,out,len(records)+1,slug,title,caption,tables,preview_only,dpi))
    def resource(items,slug,title,unit,metric,caption,lower=True,digits=3):
        fig,table=ranked_bars(items,unit,metric,lower,digits,slug);resources.extend(table)
        save_csv(out/'source_data'/(slug+'.csv'),table)
        save(fig,slug,title,caption+ ' Linear bars start at zero, with best first, length=value/best×100 and raw values on the right.',
             [slug+'.csv','resource_rankings.csv'])
    save(grouped_scene(t),'final_channels','PAMF recognition across channel conditions',
        'Final T1-E50 (13L+37U), raw or with fixed DURA trained on L or L/U. F0, 168000 physical packets in each view. '+FIXED_G,
        ['core.csv','main.csv'])
    for view,slug,title,limits in [(SIX,'DURA_protocol_gain','Restoration gain across data contracts',(-1,11)),
        ('clean','DURA_clean_change','Clean accuracy change after restoration',(-1.2,1.2))]:
        rs=[one(t,'main_effects',protocol=p,effect='native_cric_G_minus_raw',readout='DURA-LU',view=view,**COMMON) for p in PROTOCOLS]
        save(forest(PROTOCOLS,rs,'Accuracy change (percentage points)',limits),slug,title,
            'DURA-LU minus raw for final native T1. Effects are paired within F seed. Contracts have different populations and class counts, so no pooled overall average is formed. '+FIXED_G,['main_effects.csv'])
    save(matrix([[one(t,'main',protocol=p,regime='native_cric',readout='DURA-LU',view=v,**COMMON)['mean'] for v in VIEWS]
                 for p in PROTOCOLS],PROTOCOLS,SCENES,height=145),'protocol_channels','PAMF performance across data contracts',
        'Means over three F seeds. ManyRx has ten classes, other contracts six. Source-count conditions have six target RX and 144000 packets per view; F0 has seven target RX and 168000. '+FIXED_G,['main.csv'])
    for effect,slug,title in [('native_minus_supervised','T1_minus_Sup','Native T1 versus supervised training'),
                             ('native_minus_ordinaryPL','T1_minus_PL','Native T1 versus ordinary pseudo-labeling')]:
        rs=[one(t,'main_effects',protocol=p,effect=effect,readout='DURA-LU',view=SIX,**COMMON) for p in PROTOCOLS]
        save(forest(PROTOCOLS,rs,'Accuracy difference (percentage points)',(-2,5)),slug,title,
            'Same-contract, same-F-seed differences with G-LU. Six scenarios are averaged within seed before comparing. These compare complete registered recipes, not isolated SSL components. '+FIXED_G,['main_effects.csv'])
    for budget in ('natural','fixed'):
        for readout in ('raw','DURA-LU'):
            save(source_curve(t,budget,readout),'source_'+budget+'_'+readout,
                'Source receiver scaling ('+budget+' budget, '+('raw' if readout=='raw' else 'DURA')+')',
                'Only registered source counts 1,2,3,4,6 are shown, at actual numeric x positions. All use the same six target RX and 144000 packets per scene; F0 is excluded. Mean ± sample SD across three F seeds. Natural/fixed training budgets are separate experiments.', ['main.csv'])
    cross=rows(t,'crossed',test='main',view=SIX,metric='accuracy')
    labels=['Fixed G '+str(i+1) for i in range(3)]+['Independent F/G pairs']
    cross=sorted(cross,key=lambda r:(r['summary_type']!='fixed_G_over3F',str(r['G_seed'])))
    save(forest(labels,cross,'Mean channel accuracy (%)',(69,76),highlight=(3,)),
        'crossed_seeds','Recognizer and restorer seed robustness',
        'Only m3-natural: full 3×3 F/G cross. First three rows summarize three F at fixed G; the final row is the three joint diagonal F/G seeds. Nine cells are not nine independent system repetitions.',['crossed.csv'])
    bits=['111','011','101','110','000','001','010','100']
    readouts=['raw','DURA-L','DURA-LU']
    cells=[[one(t,'core',bits=b,readout=r,view=SIX,**COMMON) for r in readouts] for b in bits]
    save(matrix([[r['mean'] for r in line] for line in cells],bits,['Raw','G-L','G-LU'],
        sds=[[r['sd'] for r in line] for line in cells],height=127),'core_factorial','Recognition across the three mechanism groups',
        'A=CRIC; B=paired stochastic KL; C=native selective SSL. All eight combinations keep E50 and matched supervised CE dose. 000 is not the F15 CE recipe. Mean ± sample SD; fixed G per role.',['core.csv'])
    effects=['A_at_other_groups_on','B_at_other_groups_on','C_at_other_groups_on']
    conditions=[('raw','clean'),('raw',SIX),('DURA-L',SIX),('DURA-LU',SIX)]
    cells=[[one(t,'core_effects',effect=e,readout=r,view=v,**COMMON) for r,v in conditions] for e in effects]
    save(matrix([[r['mean'] for r in line] for line in cells],['A: CRIC','B: paired KL','C: selective SSL'],
        ['Raw\nclean','Raw\nchannels','G-L\nchannels','G-LU\nchannels'],unit='Conditional contribution (percentage points)',
        signed=True,sds=[[r['sd'] for r in line] for line in cells],height=95),'core_contributions','Conditional contributions of the three mechanism groups',
        '111 minus the corresponding one-group-off condition, paired within F seed. Positive favors the complete recipe. SD is not a confidence interval; no significance claim.',['core_effects.csv'])
    arms=[f'F{i:02d}' for i in range(1,22)]
    labels=[a+' '+label for a,label in zip(arms,FINE_NAMES)]
    for readout,view,slug,title,limits in [('raw','clean','fine_raw_clean','Fine-grained effects on clean accuracy',(-3,4)),
        ('raw',SIX,'fine_raw_channels','Fine-grained effects on channel robustness',(-4,34)),
        ('DURA-LU',SIX,'fine_DURA_channels','Fine-grained effects with DURA',(-3,6))]:
        rs=[one(t,'fine_effects',contrast='T1_minus_variant',arm=a,readout=readout,view=view,**COMMON) for a in arms]
        save(forest(labels,rs,'T1 minus variant (percentage points)',limits,height=145),slug,title,
            'Only corrected binding-m63 results. All 21 variants and negative effects retained. F15 changes a complete recipe; F05 changes stochastic forwards/compute; F19 changes CE dose and compute. They are not isolated regularizer effects. '+FIXED_G,['fine_effects.csv','fine.csv'])
    cric_effects=['CRIC_at_raw','CRIC_at_G','G_at_T1','G_at_F16','CRIC_x_G']
    rs=[one(t,'cric_G',effect=e,view=SIX,**COMMON) for e in cric_effects]
    save(forest(['CRIC at raw','CRIC with G','G with T1','G without CRIC','CRIC × G'],rs,
        'Accuracy difference (percentage points)',(-3,9)), 'CRIC_G','Matched effects of CRIC and restoration',
        'Current T1 versus corrected F16, raw/G-LU. Interaction=(T1_G−T1_raw)−(F16_G−F16_raw). The three F seeds share the same G. No stable positive interaction is established.',['cric_G.csv'])
    for readout,view,slug,title,limits in [('raw',SIX,'physics_raw_channels','Strictly supervised structural comparison',(25,60)),
        ('DURA-L',SIX,'physics_GL_channels','Strictly supervised structures with DURA',(68,79)),
        ('raw','clean','physics_raw_clean','Clean accuracy of supervised structures',(74,87))]:
        rs=[one(t,'physics',architecture=a,readout=readout,test='main',view=view,metric='accuracy') for a in ARCHS]
        save(forest([ARCH_NAMES[a] for a in ARCHS],rs,'Accuracy (%)',limits,highlight=(0,)),slug,title,
            'Nine architectures: eight new structural variants×three seeds plus reused X3_CE baseline×three. Matched L-only CE+sat.68, LS0, no U/EMA/CRIC/RDrop; E50/11100 updates. G-L has an L-only ancestry. The technical recovery mapping is already applied. This is separate from the native T1 recipe.',['physics.csv','physics_seed.csv'])
    vals=[[100*one(t,'physics_interventions',architecture=a,intervention=i,metric='default_head_flip_fraction')['mean']
            for i in INTERVENTIONS] for a in ARCHS]
    save(matrix(vals,[ARCH_NAMES[a] for a in ARCHS],INT_NAMES,unit='Default-head prediction flips (%)',height=128),
        'source_sensitivity','Prediction sensitivity under controlled source interventions',
        'Same 720 labeled source packets, 90 TX/RX/day strata, three F seeds, no target/U or fitted probe. Synthetic PA changes are not assumed identity-preserving; this is sensitivity evidence, not hardware disentanglement or calibrated channel invariance.',['physics_interventions.csv'])
    lon_conditions=[('X3_CE','raw'),('X3_CE','DURA-L'),('T1_SUP','raw'),('T1_SUP','DURA-L')]
    # Both arm names are retained from the source rather than relabeling the supervised recipe as pure CE.
    lon_key='architecture' if 'architecture' in t['Lonly'][0] else 'arm'
    if lon_key not in t['Lonly'][0]:lon_key='recipe'
    vals=[[one(t,'Lonly',**{lon_key:a},readout=r,view=v,**COMMON)['mean'] for v in VIEWS] for a,r in lon_conditions]
    save(matrix(vals,['CE, raw','CE + G-L','T1 Sup, raw','T1 Sup + G-L'],SCENES,height=103),
        'strict_Lonly','Fully supervised PAMF recipes across channel conditions',
        'F15 CE and F20 supervised T1, each three F seeds and one shared all-L G. Both have zero U exposure, but their augmentations, losses and L-forward doses differ (384 versus 768 per update). Recipe differences are not a single-component causal effect.',['Lonly.csv'])
    g_arms=list(G_NAMES)
    rs=[dict(mean=float(one(t,'G_summary',arm=a,view='six_view_mean')['accuracy_mean']),
             sd=float(one(t,'G_summary',arm=a,view='six_view_mean')['accuracy_sd'])) for a in g_arms]
    save(forest([G_NAMES[a] for a in g_arms],rs,'Mean channel accuracy (%)',(65,82),highlight=(1,)),
        'G_components','Recognition accuracy of restoration variants',
        'Seven student settings×three paired F/G seeds. Default is one existing fixed G paired with three F. This does not estimate pure G variance or complete a three-seed copy×KD factorial. All scenarios and negative results retained.',['G_summary.csv','G_seed.csv'])
    contrasts=['copy_effect_without_KD','KD_effect_from_random']
    rs=[dict(mean=float(one(t,'G_init_KD',contrast=a,view='six_view_mean')['mean_pp']),
             sd=float(one(t,'G_init_KD',contrast=a,view='six_view_mean')['sample_sd_pp'])) for a in contrasts]
    save(forest(['Copy effect without KD','KD from random initialization'],rs,'Accuracy difference (percentage points)',(-1,8)),
        'G_initialization_KD','Matched initialization and distillation contrasts',
        'Copy_no_KD−random_no_KD and random_with_KD−random_no_KD, paired across three F/G seeds. Missing three-seed copy+KD cells are not fabricated; no complete interaction is claimed.',['G_init_KD.csv'])
    for readout,title in [('raw','Recognition of external Full and Full + PL methods'),
                          ('DURA-LU','External methods and PAMF after restoration')]:
        save(external_f0(t,readout),'external_F0_'+readout,title,
            'F0, same physical test packets and seven scenes. Six scenes averaged within each F seed, then mean ± sample SD. Full/Full+PL are the original external recipes. Current PAMF is native T1, not the ordinary-PL control. All G-LU readouts use the contract-matched frozen DURA. Training recipes differ.',['external_full.csv','external_full_seed.csv','main.csv'])
    save(external_f0(t,'DURA-L',True),'external_F0_Lonly','Fully supervised external systems and PAMF',
        'F0 Full supervised F+G-L only; both F and all G ancestors use L. PAMF here is the matched supervised control, not native T1 or F20 supervised regularization. Source/target contract and G-L IQ function are shared. Three F seeds condition on one G.',['external_full.csv','gfactor.csv'])
    ext_labels=[FAM_NAMES[f]+(' Full' if r=='supervised' else ' Full + PL') for f in FAMILIES for r in ('supervised','pl')]
    ext_values=[]
    for f in FAMILIES:
        for regime in ('supervised','pl'):
            line=[]
            for p in PROTOCOLS:
                rs=[r for r in ext_effects if r['protocol']==p and r['family']==f and r['regime']==regime
                    and r['test']=='main' and r['view']==SIX and r['metric']=='accuracy' and r['effect']=='G-LU_minus_raw']
                line.append(rs[0]['mean'] if len(rs)==1 else math.nan)
            ext_values.append(line)
    # Compact numeric contract IDs leave readable x labels; their mapping is in source captions and the report.
    save(matrix(ext_values,ext_labels,[str(i+1) for i in range(17)],unit='Accuracy gain (percentage points)',
        limits=(0,60),height=130),'external_DURA_gain','Restoration gains for external recognition methods',
        'Column IDs follow the ordered protocol list in source_data/snapshot.json: '+', '.join(f'{i+1}={p}' for i,p in enumerate(PROTOCOLS))+'. '
        'All 86 registered Full/Full+PL main-task conditions are retained. N/A means that family/regime/contract was not registered, not a zero score. Values are paired six-scene gains from frozen scores.',['external_full_paired_effects.csv'])
    fair_cells=['Sup+G-L','PL+G-L','Sup+G-LU','PL+G-LU']
    save(matrix([[one(t,'gfactor',protocol_id=p,cell=c,test='main',view=SIX,metric='accuracy')['mean']
         for c in fair_cells] for p in PROTOCOLS],PROTOCOLS,['Sup\nG-L','PL\nG-L','Sup\nG-LU','PL\nG-LU'],height=145),
        'fairness_cells','PAMF performance under labeled and unlabeled training',
        'Complete 17 contracts×three F seeds×four cells. Only Sup+G-L is fully L-only; PL+G-L uses U through F. Native T1 is not substituted into this ordinary-PL factorial. Means only; paired sample SD is retained in the accompanying effects table.',['gfactor.csv'])
    fair_effects=['PL_minus_Sup_at_G_L','PL_minus_Sup_at_G_LU','G_LU_minus_L_at_Sup','G_LU_minus_L_at_PL','F_PL_x_G_U_interaction']
    save(matrix([[one(t,'gfactor_effects',protocol_id=p,effect=e,test='main',view=SIX,metric='accuracy')['mean']
        for e in fair_effects] for p in PROTOCOLS],PROTOCOLS,
        ['PL effect\nG-L','PL effect\nG-LU','U in G\nSup F','U in G\nPL F','F × G\ninteraction'],
        unit='Paired accuracy difference (percentage points)',signed=True,height=145),'fairness_effects','Paired effects of unlabeled source data',
        'All effects calculated within F seed after six-scenario averaging. Negative values retained. The zero-centered scale uses the same units in every cell. Sample SD is not a significance test.',['gfactor_effects.csv'])
    rs=[one(t,'gfactor_effects',protocol_id=p,effect='F_PL_x_G_U_interaction',test='main',view=SIX,metric='accuracy') for p in PROTOCOLS]
    save(forest(PROTOCOLS,rs,'Interaction (percentage points)',(-1.5,1.5)),
        'fairness_interaction','Interaction between pseudo-labeling and restorer training',
        'Difference in differences: (PL_G-LU−PL_G-L)−(Sup_G-LU−Sup_G-L). Three paired F seeds and fixed G per role. Intervals are sample SD, not confidence intervals; no universal positive synergy is supported.',['gfactor_effects.csv'])
    unified={(r['case'],int(r['batch'])):r for r in t['unified_timing']}
    for metric,batch,unit,factor,lower,slug,title in [('median_ms',1,'ms',1,True,'unified_latency','Single-sample inference latency'),
        ('throughput_IQ_s',128,'kIQ/s',.001,False,'unified_throughput','Batch-128 inference throughput')]:
        cases=[f+'_full' for f in FAMILIES]+['T4_graph_reuse_GF'];names=[FAM_NAMES[f] for f in FAMILIES]+['PAMF']
        resource([dict(label=name,value=float(unified[c,batch][metric])*factor,color=COLORS['pamf'] if name=='PAMF' else COLORS[FAMILIES[i]])
            for i,(c,name) in enumerate(zip(cases,names))],slug,title,unit,
            'Median latency' if lower else 'Throughput',
            'Unified latency-r02 cohort, exclusive RTX3090, FP32/TF32off. Five timing rounds, fixed seed01 weights, GPU-wall timing excluding transfer. External cases are standalone Full recognizers without G; PAMF includes its complete G+F. This is not a matched G+F speed comparison for restored external methods. The cohort is separate from later mechanism/structure benchmarks.',lower)
    fine={(r['case'],int(r['batch'])):r for r in t['fine_timing']}
    routes=['T0_native_GF','T1_tensor_GF','T2_reuse_GF','T3_graph_GF','T4_graph_reuse_GF']
    resource([dict(label=label,value=fine[c,1]['median_ms'],color=COLORS['pamf'] if c==routes[-1] else COLORS['baseline'])
        for c,label in zip(routes,['Native','Tensor','Reuse','Graph','Graph + reuse'])],
        'execution_latency','Latency across PAMF execution paths','ms','Median latency',
        'Dedicated fine-timing cohort, B1, pooled median of five rounds. Same T1/DURA weights and FP32 precision. Full-data route prediction equivalence is not inferred from timing alone.')
    fcases=['T4_graph_reuse_GF']+[f'F{i:02d}_GF' for i in range(6,11)]
    resource([dict(label='PAMF' if c==fcases[0] else FINE_NAMES[int(c[1:3])-1],value=fine[c,1]['median_ms'],
        color=COLORS['pamf'] if c==fcases[0] else COLORS['baseline']) for c in fcases],
        'F_mechanism_latency','Inference latency of recognizer variants','ms','Median latency',
        'Dedicated fine-timing cohort, B1. Every listed F variant uses the same optimized graph+reuse G/F engine. A/B/C training-only mechanisms have zero added frozen deployment operators, not zero training cost.')
    gcases=['T0_native_GF']+[a+'_GF' for a in list(G_NAMES)[2:]]
    resource([dict(label='DURA, native' if c==gcases[0] else G_NAMES[c.removesuffix('_GF')],
        value=fine[c,1]['median_ms'],color=COLORS['pamf'] if c==gcases[0] else COLORS['baseline']) for c in gcases],
        'G_mechanism_latency','Native inference latency of restoration variants','ms','Median latency',
        'Dedicated fine-timing cohort, B1. G-student cases use their native eager path with the same T1 classifier. Native default is the reference; the optimized default is intentionally not used as the comparator. Python wrapper overhead is included. Architectural accuracy and parameter differences must be considered separately.')
    st={(r['case'],int(r['batch'])):r for r in t['structure_timing']}
    resource([dict(label=ARCH_NAMES[a],value=st[a+'_GF',1]['median_ms'],
        color=COLORS['pamf'] if a=='X3_CE' else COLORS['baseline']) for a in ARCHS],
        'structure_latency','Inference latency of supervised structures','ms','Median latency',
        'Dedicated strict-supervised structure cohort, B1. Identical graph+reuse engine and fixed G-L; matched CE structure weights. Five timing rounds are not five model seeds. Timing values are not ranked against other cohorts.')
    profiles=data['metadata']['timing_profiles']['structure_timing'];params=profiles['mechanism_parameters']
    resource([dict(label=ARCH_NAMES[a],value=(params[a]+profiles['G_parameters'])/1e6,
        color=COLORS['pamf'] if a=='X3_CE' else COLORS['baseline']) for a in ARCHS],
        'structure_parameters','System parameter counts of supervised structures','M','Parameters',
        'Full registered F+G parameter counts, including default classifier and the entire 266951-parameter G-L. Full X3-CE has 499944 parameters. These are parameter counts, not serialized file size, FLOPs or measured memory.',True,3)
    geom_conditions=['111','011','101','110']
    values=[[float(one(t,'core_geometry',bits=b,source_metric=metric)['mean']) for metric in
        ('same_TX_cross_RX_distance','different_TX_same_RX_distance','separation_margin')] for b in geom_conditions]
    save(matrix(np.array(values)*100,geom_conditions,['Same TX\ncross RX','Different TX\nsame RX','Separation\nmargin'],
        unit='Cosine distance × 100',height=95,limits=(0,60)),
        'source_geometry','Source feature geometry of the core mechanisms',
        'Same 720 L-only source observations and anchor-balanced, same-day pairs, averaged over three F seeds. Distances multiplied by 100 for display only. A smaller same-TX distance alone does not establish improved identity separation or full RX/channel invariance.',['core_geometry.csv'])
    focused=arms[:14]
    rs=[one(t,'fine_effects',contrast='T1_minus_variant',arm=a,readout='raw',view=SIX,**COMMON) for a in focused]
    save(forest([a+' '+label for a,label in zip(focused,FINE_NAMES[:14])],
        rs,'T1 minus variant (percentage points)',(-2,9),height=125),
        'fine_feature_consistency','Feature and consistency effects on raw robustness',
        'Predefined F01–F14 feature/consistency diagnostics, shown separately from F15–F21 recipe/CE-dose changes so small differences are readable. The full F01–F21 panel remains Fig. 15; no rows are selected by performance. Mean ± paired sample SD.',['fine_effects.csv'])
    save_csv(out/'source_data/resource_rankings.csv',resources)
    write(out/('draft_figures.json' if preview_only else 'figure_manifest.json'),
        dict(status='PREVIEW' if preview_only else 'VERIFIED',figures=records,figure_count=len(records),
             font='Times New Roman',dpi=dpi,one_data_chart_per_image=True,no_subtitles_or_footnotes=True))
    if not preview_only:
        writer=PdfWriter()
        for r in records:writer.append(str(out/'pdf'/(r['stem']+'.pdf')))
        with (out/'all_figures.pdf').open('wb') as stream:writer.write(stream)
        captions=['# Figure captions','',FIXED_G,'']
        for r in records:captions += [f'## Fig. {r["number"]}. {r["title"]}','',r['caption'],'','Source tables: '+', '.join(r['source_tables'])+'.','']
        (out/'captions_en.md').write_text('\n'.join(captions),encoding='utf-8')
        save_csv(out/'figure_index.csv',[dict(number=r['number'],stem=r['stem'],title=r['title']) for r in records])
        reproduce=out/'reproduce';reproduce.mkdir(exist_ok=True)
        for filename in ('collect_completed_pamf.py','analyze_completed_pamf.py','plot_completed_pamf.py',
                         'paper_publication_style.py','plot_pamf_paper_figures.py'):
            shutil.copyfile(Path(__file__).with_name(filename),reproduce/filename)
    print(json.dumps(dict(status='PREVIEW_COMPLETE' if preview_only else 'FIGURES_VERIFIED',figures=len(records),dpi=dpi)),flush=True)
    return records


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--dpi',type=int,default=4800)
    parser.add_argument('--preview-only',action='store_true');args=parser.parse_args()
    if (args.output/'figure_manifest.json').exists():raise FileExistsError('Preserve completed figures')
    apply_style();render(args.source,args.output,args.preview_only,args.dpi)
