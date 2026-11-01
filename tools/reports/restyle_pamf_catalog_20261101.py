"""Restyle the frozen PAMF chart catalog without changing its statistics or series colors."""
from __future__ import annotations

import argparse
import ctypes
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import time
import warnings

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

def load_renderer(path: Path):
    spec=importlib.util.spec_from_file_location('pamf_r3_render',path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def configure_renderer(module,style):
    base=module.panel_render
    ink='#000000' if style=='paper' else '#21364B'
    light='#F6F1E7'; deep='#15263B'; plot='#FFFFFF'
    # Preserve every experimental series color from the historical renderer.
    plt.rcParams.update({'figure.facecolor':deep if style=='space' else 'white',
        'savefig.facecolor':deep if style=='space' else 'white',
        'axes.facecolor':plot,'axes.labelcolor':light if style=='space' else ink,
        'axes.titlecolor':light if style=='space' else ink,
        'xtick.color':light if style=='space' else ink,
        'ytick.color':light if style=='space' else ink,
        'legend.labelcolor':light if style=='space' else ink,
        'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],
        'svg.fonttype':'none','pdf.fonttype':42})
    def panel(ax,p,index):
        base(ax,p,index)
        if p['kind']=='heatmap' and ax.images:
            # imshow embeds pixels in SVG; replace only the display primitive
            # with identically normalized vector cells. Values and color map stay fixed.
            im=ax.images[0];array=im.get_array();height,width=array.shape
            colorbar=getattr(im,'colorbar',None)
            if colorbar is not None and colorbar.solids is not None:
                colorbar.solids.set_rasterized(False)
            ax.pcolormesh(np.arange(width+1)-.5,np.arange(height+1)-.5,array,
                          cmap=im.cmap,norm=im.norm,shading='flat',rasterized=False,
                          zorder=im.get_zorder())
            im.remove();ax.set_xlim(-.5,width-.5);ax.set_ylim(height-.5,-.5)
        for spine in ax.spines.values():spine.set_color(ink)
        ax.set_facecolor(plot)
        for grid in ax.get_xgridlines()+ax.get_ygridlines():
            if style=='space':grid.set_color('#DAE3EA')
            grid.set_linewidth(.55)
        ax.xaxis.label.set_fontsize(max(9.3,ax.xaxis.label.get_fontsize()))
        ax.yaxis.label.set_fontsize(max(9.3,ax.yaxis.label.get_fontsize()))
        ax.title.set_color(light if style=='space' else ink)
        ax._left_title.set_color(light if style=='space' else ink)
        ax._right_title.set_color(light if style=='space' else ink)
        for tick in ax.get_xticklabels()+ax.get_yticklabels():
            tick.set_fontsize(max(8.2,tick.get_fontsize()))
        legend=ax.get_legend()
        if legend:
            for t in legend.get_texts():
                t.set_fontsize(max(8.1,t.get_fontsize()))
                t.set_color(light if style=='space' else ink)
        # Panel letters sit just outside the axes on the dark frame. Data labels
        # inside the white plot retain their original color, including series colors.
        for t in ax.texts:
            if t.get_transform()==ax.transAxes and t.get_text() in list('abcdefgh'):
                t.set_color(light if style=='space' else ink)
        if style=='space':
            # The parameters-axis break sits outside the white data area.
            if p.get('parameter_break'):
                for line in ax.lines:
                    if not line.get_clip_on():line.set_color(light)
            for extra in ax.figure.axes:
                if extra is ax:continue
                extra.tick_params(colors=light)
                extra.xaxis.label.set_color(light);extra.yaxis.label.set_color(light)
    module.panel_render=panel

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--theme',choices=['paper','space'],required=True)
    p.add_argument('--selected-first',action='store_true')
    p.add_argument('--force',action='store_true')
    p.add_argument('--ids',nargs='*')
    p.add_argument('--kind')
    args=p.parse_args()
    try:ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(),0x00004000)
    except AttributeError:pass
    catalog=json.loads((args.source/'catalog.json').read_text(encoding='utf-8'))
    dest=args.output/args.theme
    for name in ['pdf','svg','png','source_data']:(dest/name).mkdir(parents=True,exist_ok=True)
    shutil.copy2(args.source/'catalog.json',dest/'catalog.json')
    module=load_renderer(args.source/'reproduce'/'render_figures.py')
    configure_renderer(module,args.theme)
    figs=catalog['figures']
    if args.selected_first:figs=sorted(figs,key=lambda f:not f['selected'])
    if args.ids:figs=[f for f in figs if f['id'] in args.ids]
    if args.kind:figs=[f for f in figs if any(p['kind']==args.kind for p in f['panels'])]
    log=[];start=time.time()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        for i,f in enumerate(figs,1):
            expected=[dest/ext/(f['id']+'.'+ext) for ext in ['svg','pdf','png']]
            if not args.force and all(x.exists() for x in expected):
                print(f'SKIP {args.theme} {i}/{len(figs)} {f["id"]}',flush=True)
                continue
            result=module.render(f,dest);log.append(result)
            print(f'RENDER {args.theme} {i}/{len(figs)} {f["id"]}',flush=True)
    report={'theme':args.theme,'requested':len(figs),'newly_rendered':len(log),
        'source_catalog':str(args.source/'catalog.json'),'series_colors':'unchanged',
        'statistics':'unchanged','elapsed_seconds':round(time.time()-start,1),
        'warnings':[str(w.message) for w in caught]}
    (dest/'render_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False),flush=True)

if __name__=='__main__':main()
