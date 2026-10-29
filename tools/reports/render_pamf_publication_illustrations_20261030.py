"""Editable vector illustrations of the frozen PAMF/DURA methods.

All waveforms and landscapes here are schematic illustrations, not measured data.
The diagram topology follows PAMF_METHOD_DEFINITION_20261027.md and DURA_V1.md.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path as FilePath

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import (Arc, Circle, Ellipse, FancyArrowPatch,
                                PathPatch, Polygon, Rectangle, Wedge)
from matplotlib.path import Path
import numpy as np

mpl.rcParams.update({'font.family':'DejaVu Sans','font.size':7.2,
                     'svg.fonttype':'none','pdf.fonttype':42,
                     'savefig.facecolor':'white','axes.linewidth':0.6})

C={'ink':'#263e4f','muted':'#627785','faint':'#D8E1E4','grid':'#E7EDEF',
   'blue':'#577e9d','blue_light':'#d7e6ee','teal':'#5e9996',
   'teal_light':'#d5eae7','rose':'#bc4865','rose_light':'#f0dfe5',
   'gold':'#d89a38','gold_light':'#f7ead1','violet':'#897ba0',
   'violet_light':'#e8e3ef','earth':'#e9f1ed','paper':'#ffffff'}
QA=[]

def canvas(h=4.65):
    fig=plt.figure(figsize=(7.2,h),facecolor='white')
    ax=fig.add_axes([0,0,1,1]); ax.set_xlim(0,120); ax.set_ylim(0,70)
    ax.axis('off'); return fig,ax

def txt(ax,x,y,s,size=7.1,color='ink',weight='normal',ha='left',va='center',**kw):
    ax.text(x,y,s,fontsize=size,color=C.get(color,color),weight=weight,
            ha=ha,va=va,linespacing=1.22,**kw)

def path(ax,pts,color='muted',lw=.85,alpha=1,ls='-',z=3):
    ax.plot([p[0] for p in pts],[p[1] for p in pts],color=C.get(color,color),
            lw=lw,alpha=alpha,ls=ls,solid_capstyle='round',zorder=z)

def arrow(ax,start,end,color='muted',lw=.9,rad=0,scale=6,z=6,ls='-'):
    p=FancyArrowPatch(start,end,connectionstyle=f'arc3,rad={rad}',arrowstyle='-|>',
        mutation_scale=scale,linewidth=lw,color=C.get(color,color),linestyle=ls,zorder=z)
    ax.add_patch(p)

def ribbon(ax,p0,p1,width,color,alpha=.24,curve=.0,z=1):
    x0,y0=p0; x1,y1=p1; dx=x1-x0
    verts=[(x0,y0-width/2),(x0+dx*.42,y0-width/2+curve),
           (x1-dx*.42,y1-width/2+curve),(x1,y1-width/2),
           (x1,y1+width/2),(x1-dx*.42,y1+width/2+curve),
           (x0+dx*.42,y0+width/2+curve),(x0,y0+width/2),(x0,y0-width/2)]
    codes=[Path.MOVETO,Path.CURVE4,Path.CURVE4,Path.CURVE4,Path.LINETO,
           Path.CURVE4,Path.CURVE4,Path.CURVE4,Path.CLOSEPOLY]
    ax.add_patch(PathPatch(Path(verts,codes),fc=C[color],ec='none',alpha=alpha,zorder=z))

def waveform(ax,x,y,w,h,color='blue',kind='iq',seed=12,alpha=1,lw=.9):
    r=np.random.default_rng(seed); t=np.linspace(0,1,170)
    envelope=.45+.55*np.exp(-((t-.48)/.33)**2)
    if kind=='iq':
        v=envelope*(.42*np.sin(2*np.pi*(7*t+1.7*t*t))+.21*np.sin(2*np.pi*(24*t+.08)))
        v+=r.normal(0,.075,len(t))
    elif kind=='restored':
        v=envelope*(.51*np.sin(2*np.pi*(7*t+1.7*t*t))+.16*np.sin(2*np.pi*24*t))
        v+=r.normal(0,.035,len(t))
    else:
        v=np.exp(-5*t)*np.cos(2*np.pi*8*t)
    path(ax,[(x,y),(x+w,y)],'grid',.55,z=1)
    ax.plot(x+t*w,y+v*h,color=C[color],lw=lw,alpha=alpha,zorder=3,solid_capstyle='round')

def spectrum(ax,x,y,w,h):
    t=np.linspace(-1,1,140)
    p=np.exp(-((t-.39)/.17)**2)+.7*np.exp(-((t+.39)/.17)**2)
    ax.fill_between(x+(t+1)*w/2,y,y+h*p,where=t<0,color=C['teal_light'],lw=0,zorder=2)
    ax.fill_between(x+(t+1)*w/2,y,y+h*p,where=t>=0,color=C['teal'],alpha=.24,lw=0,zorder=2)
    ax.plot(x+(t+1)*w/2,y+h*p,color=C['teal'],lw=.9,zorder=3)
    path(ax,[(x,y),(x+w,y)],'faint',.55)
    path(ax,[(x+w/2,y),(x+w/2,y+h*1.05)],'faint',.55,ls=':')

def polynomial(ax,x,y,w,h):
    t=np.linspace(-1,1,90)
    for k,c in [(1,'gold'),(.65,'violet')]:
        v=t+k*.42*t**3
        ax.plot(x+(t+1)*w/2,y+(v+1.42)*h/2.85,color=C[c],lw=.9,zorder=3)
    path(ax,[(x,y+h/2),(x+w,y+h/2)],'faint',.5)

def response(ax,x,y,w,h):
    t=np.linspace(0,1,130)
    f=np.exp(-3.2*t)*np.cos(2*np.pi*3.3*t)
    g=.73*np.exp(-2.7*t)*np.cos(2*np.pi*3.3*t+.25)
    ax.plot(x+t*w,y+h*f,color=C['violet'],lw=.95)
    ax.plot(x+t*w,y+h*g,color=C['muted'],lw=.65,ls='--')

def bars(ax,x,y,w,h,color='blue',n=6):
    for i in range(n):
        yi=y+i*h/n
        ax.add_patch(Polygon([(x,yi),(x+w,yi+.5),(x+w,yi+1.1),(x,yi+.6)],
            closed=True,fc=C[color],ec='none',alpha=.15+i*.055,zorder=2))

def stack(ax,x,y,w,h,color='blue',layers=4):
    for i in range(layers):
        yy=y+i*h*.19; dx=i*.55
        poly=Polygon([(x+dx,yy),(x+w+dx,yy),(x+w+dx+1.9,yy+1.35),(x+dx+1.9,yy+1.35)],
                     closed=True,facecolor=C[color],edgecolor=C[color],lw=.45,alpha=.18+i*.08,zorder=2+i)
        ax.add_patch(poly)

def sig_trajectories(ax,x,y,w,h):
    bases=[np.array([.18,.30,.19,.63,.45,.78,.66,.86]),
           np.array([.23,.18,.44,.32,.66,.55,.81,.76]),
           np.array([.08,.35,.27,.49,.39,.68,.60,.91])]
    for j,(a,c) in enumerate(zip(bases,['blue','teal','rose'])):
        xx=x+np.linspace(0,w,8); yy=y+j*h*.22+a*h*.4
        ax.plot(xx,yy,color=C[c],lw=1.0,zorder=4)
        ax.scatter(xx,yy,s=2.8,color=C[c],zorder=5)

def fusion(ax,x,y,scale=1):
    r=5.0*scale
    # The temporal and spectral streams form one f_TF; the other three
    # residual components are f_PA, f_R and f_SIG, hence four sectors.
    for i,col in enumerate(['teal','gold','violet','rose']):
        ax.add_patch(Wedge((x,y),r,90+i*90,90+(i+1)*90-3,width=1.4*scale,
                          facecolor=C[col],edgecolor='white',lw=.35,zorder=6))
    txt(ax,x,y,'FUSE',6.1*scale,'ink','bold','center')

def head(ax,x,y):
    # Qualitative classifier scores: not experimental measurements.
    for i,(c,length) in enumerate(zip(['blue','rose','gold'],[5.4,3.6,2.5])):
        yy=y+4-i*4
        ax.add_patch(Circle((x+1,yy),.65,fc=C[c],ec='none',zorder=4))
        path(ax,[(x+2,yy),(x+2+length,yy)],c,1.8,alpha=.75,z=4)
    txt(ax,x+4,y-8,'identity',6.5,'muted',ha='center')

def top(ax,title,kicker):
    txt(ax,3,67,kicker.upper(),6.1,'muted','bold')
    txt(ax,3,61.7,title,9.2,'ink','bold')
    path(ax,[(3,58.7),(117,58.7)],'grid',.65)

def save(fig,out,name):
    out.mkdir(parents=True,exist_ok=True)
    fig.canvas.draw()
    renderer=fig.canvas.get_renderer(); page=fig.bbox
    clipped=[]; texts=[]
    for artist in fig.findobj(match=mpl.text.Text):
        if not artist.get_visible() or not artist.get_text().strip():continue
        b=artist.get_window_extent(renderer)
        texts.append((artist.get_text(),b))
        if b.x0 < page.x0-0.5 or b.y0 < page.y0-0.5 or b.x1 > page.x1+0.5 or b.y1 > page.y1+0.5:
            clipped.append(artist.get_text())
    overlaps=[]
    for i,(a,ba) in enumerate(texts):
        for b,bb in texts[i+1:]:
            if min(ba.x1,bb.x1)-max(ba.x0,bb.x0)>1 and min(ba.y1,bb.y1)-max(ba.y0,bb.y0)>1:
                overlaps.append([a,b])
    QA.append({'figure':name,'text_outside_canvas':clipped,'text_overlaps':overlaps})
    for ext in ['svg','pdf','png']:
        fig.savefig(out/(name+'.'+ext),dpi=260 if ext=='png' else 72,
                    facecolor='white',edgecolor='none')
    plt.close(fig)

def feature_lanes(ax,x,y,width=34,gap=9.0,labels=True):
    ys=[y+3*gap,y+2*gap,y+gap,y]
    names=['Temporal encoder','Mirrored spectrum','Nonlinear memory','Reference response']
    colors=['blue','teal','gold','violet']
    for i,(yy,name,c) in enumerate(zip(ys,names,colors)):
        ax.add_patch(Circle((x,yy),1.15,fc=C[c],ec='none'))
        if labels: txt(ax,x+2.5,yy+2.0,name,6.7,'ink','bold')
        vx=x+18
        if i==0:waveform(ax,vx,yy-1,width-18,3,'blue',seed=13)
        if i==1:spectrum(ax,vx,yy-2,width-18,3.1)
        if i==2:polynomial(ax,vx,yy-3,width-18,6)
        if i==3:response(ax,vx,yy-1,width-18,2.7)
    return ys

def network_a(out):
    fig,ax=canvas(4.55);top(ax,'PAMF: complementary views of one IQ observation','network / A')
    txt(ax,3,38,'Received IQ',7.0,'ink','bold'); waveform(ax,3,33,15,5,'blue',seed=2)
    waveform(ax,3,26.5,15,4,'rose',seed=7)
    txt(ax,3,21.5,'I / Q',6.4,'muted')
    ys=feature_lanes(ax,25,16,31,9.4)
    for yy,c in zip(ys,['blue','teal','gold','violet']):ribbon(ax,(19,30),(24,yy),1.7,c,.18)
    for yy,c in zip(ys[:2],['blue','teal']):
        ribbon(ax,(58,yy),(67,39.5),1.4,c,.23)
    ax.add_patch(Circle((68,39.5),1.1,fc=C['teal'],ec='white',lw=.45,zorder=5))
    ribbon(ax,(69,39.5),(91,33),1.6,'teal',.23)
    txt(ax,70,43,'time–frequency',6.0,'muted')
    for yy,c in zip(ys[2:],['gold','violet']):
        ribbon(ax,(58,yy),(91,32),1.5,c,.23)
    txt(ax,64,53,'Three SIG-TIME views',6.8,'rose','bold')
    sig_trajectories(ax,64,45,15,6)
    arrow(ax,(57,ys[0]),(63,50),'rose',.8,rad=.2)
    ribbon(ax,(79,47),(91,35),1.5,'rose',.22)
    fusion(ax,95,32);txt(ax,95,22,'residual fusion',6.5,'ink',ha='center')
    arrow(ax,(100.5,32),(106,32),'ink',.8);head(ax,107,32)
    txt(ax,107,41,'cosine head',6.1,'muted',ha='center')
    txt(ax,3,6,'One input · fixed local feature permutation · original cosine head',6.3,'muted')
    save(fig,out,'PAMF_A_multibranch')

def network_b(out):
    fig,ax=canvas(4.85);top(ax,'PAMF: signal representations converge on identity','network / B')
    # Input waveform is the visual anchor, rather than an input box.
    txt(ax,3,44,'RECEIVED SIGNAL',6.3,'muted','bold')
    waveform(ax,3,39,21,5,'blue',seed=3);waveform(ax,3,33,21,4,'rose',seed=9)
    txt(ax,3,27,'one I/Q window',6.8,'ink')
    positions=[(37,48,'Temporal','blue'),(37,33,'Mirrored spectrum','teal'),
               (37,18,'PA-inspired','gold'),(66,18,'Reference','violet')]
    for x,y,label,c in positions:
        ribbon(ax,(25,36),(x-2,y),2.1,c,.17)
        txt(ax,x,y+5,label,6.7,'ink','bold')
        if c=='blue':waveform(ax,x,y,17,3,'blue',seed=4)
        elif c=='teal':spectrum(ax,x,y-2,17,4)
        elif c=='gold':polynomial(ax,x,y-3,17,6)
        else:response(ax,x,y,17,3)
    ribbon(ax,(55,48),(65,47),1.5,'blue',.22)
    txt(ax,65,53,'Three-view SIG-TIME',6.7,'rose','bold')
    sig_trajectories(ax,65,45,15,5)
    for start,c in [((55,48),'blue'),((55,33),'teal')]:ribbon(ax,start,(62,37),1.5,c,.22)
    ax.add_patch(Circle((63,37),1.0,fc=C['teal'],ec='white',lw=.4,zorder=5))
    ribbon(ax,(64,37),(94,33),1.7,'teal',.22)
    for start,c in [((55,18),'gold'),((84,18),'violet'),((80,47),'rose')]:
        ribbon(ax,start,(94,32),1.8,c,.22)
    # Temporal and spectral features fuse together before the final residual fusion.
    txt(ax,64,34,'time–frequency',5.8,'muted')
    fusion(ax,98,32,1.05);arrow(ax,(104,32),(108,32),'ink',.8);head(ax,108,32)
    txt(ax,108,41,'cosine head',6.0,'muted',ha='center')
    txt(ax,3,6,'SIG-TIME changes local feature traversal; the raw waveform is not reordered.',6.3,'muted')
    save(fig,out,'PAMF_B_feature_fan')

def network_c(out):
    fig,ax=canvas(5.0);top(ax,'Three-view SIG-TIME inside PAMF','network / C')
    txt(ax,3,54,'a',9,'ink','bold');txt(ax,8,54,'Shared representation',7.0,'ink','bold')
    waveform(ax,4,42,15,4,'blue',seed=3);txt(ax,4,36,'received IQ',6.3,'muted')
    stack(ax,24,39,13,6,'blue');txt(ax,24,34,'temporal feature',6.4,'muted')
    arrow(ax,(19,42),(23,42),'blue',.8)
    for i,c in enumerate(['blue','teal','rose']):
        yy=50-i*7
        ribbon(ax,(39,42),(48,yy),1.35,c,.25)
        txt(ax,50,yy,f'view {i+1}',6.4,c,'bold')
        sig_trajectories(ax,61,yy-1,13,2.2)
    txt(ax,80,48,'local path',6.3,'ink','bold');txt(ax,80,40,'increment + area',6.0,'muted')
    for yy,c in zip([50,43,36],['blue','teal','rose']):ribbon(ax,(75,yy),(93,43),1.0,c,.22)
    fusion(ax,98,43,.8);arrow(ax,(102.5,43),(107,43),'ink',.7);head(ax,107,43)
    txt(ax,108,52,'cosine head',6.0,'muted',ha='center')
    txt(ax,75,32,'time–frequency + PA + reference',5.9,'muted')
    arrow(ax,(98,34),(98,38),'muted',.7,scale=5)
    path(ax,[(3,28),(117,28)],'grid',.65)
    txt(ax,3,23,'b',9,'ink','bold');txt(ax,8,23,'Local feature traversal',7,'ink','bold')
    order=[0,5,3,6,2,1,4,7]
    xx=np.linspace(15,61,8)
    for i,x in enumerate(xx):
        ax.add_patch(Circle((x,14.5),1.1,fc=C['blue_light'],ec=C['blue'],lw=.55))
        txt(ax,x,14.5,str(i),5.5,'ink',ha='center')
        if i<7:arrow(ax,(x+1.25,14.5),(xx[i+1]-1.25,14.5),'faint',.55,scale=4)
    txt(ax,15,8,'original feature positions',6.2,'muted')
    txt(ax,68,19,'Fixed order',6.5,'rose','bold')
    txt(ax,68,13,'0  →  5  →  3  →  6  →  2  →  1  →  4  →  7',6.2,'ink')
    txt(ax,68,7,'Only the local feature path is reordered.',6.1,'muted')
    save(fig,out,'PAMF_C_signature_detail')

def network_d(out):
    fig,ax=canvas(5.35);top(ax,'PAMF architecture and three-view encoding','network / D')
    txt(ax,3,55,'a',9,'ink','bold');txt(ax,8,55,'Complementary branches',7.0,'ink','bold')
    waveform(ax,4,41,13,4,'blue',seed=3);txt(ax,4,35,'I/Q window',6.1,'muted')
    lanes=[(49,'Temporal','blue'),(43,'Spectrum','teal'),(37,'PA memory','gold'),(31,'Reference','violet')]
    for yy,label,c in lanes:
        ribbon(ax,(18,41),(23,yy),1.4,c,.2)
        txt(ax,24,yy+1.5,label,6.2,'ink','bold')
        if c=='blue':waveform(ax,40,yy,11,2.1,c,seed=3)
        elif c=='teal':spectrum(ax,40,yy-1.2,11,2.3)
        elif c=='gold':polynomial(ax,40,yy-1.5,11,3)
        else:response(ax,40,yy,11,1.9)
    for yy,c in [(49,'blue'),(43,'teal')]:ribbon(ax,(53,yy),(62,44.5),1.1,c,.2)
    ax.add_patch(Circle((63,44.5),.8,fc=C['teal'],ec='white',lw=.3))
    txt(ax,64.5,46.2,'time–frequency',5.8,'muted')
    ribbon(ax,(64,44.5),(91,39),1.3,'teal',.24)
    for yy,c in [(37,'gold'),(31,'violet')]:ribbon(ax,(53,yy),(91,39),1.3,c,.22)
    txt(ax,65,53,'SIG-TIME views',6.2,'rose','bold');sig_trajectories(ax,65,49,13,3.5)
    arrow(ax,(52,49),(64,50),'rose',.65,rad=.15,scale=4)
    ribbon(ax,(78,50),(91,41),1.2,'rose',.22)
    fusion(ax,96,39,.82);arrow(ax,(100.5,39),(106,39),'ink',.7);head(ax,107,39)
    txt(ax,106,48,'cosine head',5.9,'muted',ha='center')
    path(ax,[(3,26.5),(117,26.5)],'grid',.65)
    txt(ax,3,23,'b',9,'ink','bold');txt(ax,8,23,'One shared intermediate, three learned projections',7,'ink','bold')
    for j,c in enumerate(['blue','teal','rose']):
        yy=16-j*3.3
        path(ax,[(8,yy),(15,yy)],c,1.4);ax.add_patch(Circle((8,yy),.65,fc=C[c],ec='none'))
    txt(ax,20,14,'96 → 6\nper view',6.2,'ink')
    arrow(ax,(30,14),(36,14),'muted',.7)
    bars(ax,39,10,12,8,'blue',4);txt(ax,38,6,'4 windows / view',5.9,'muted')
    arrow(ax,(53,14),(59,14),'muted',.7)
    txt(ax,61,18,'fixed local order',6.2,'rose','bold')
    txt(ax,61,13,'0 · 5 · 3 · 6 · 2 · 1 · 4 · 7',6.0,'ink')
    txt(ax,61,7,'IQ order is unchanged',5.8,'muted')
    arrow(ax,(90,14),(95,14),'muted',.7)
    txt(ax,96,19,'increment + area',6.1,'ink','bold')
    txt(ax,96,13,'7 + 21 per window',5.9,'muted')
    txt(ax,96,7,'concatenate → feature',5.9,'muted')
    save(fig,out,'PAMF_D_architecture_detail')

def stage_glyph(ax,x,y,number):
    # A visual stack: learned prior and data-consistency solve are distinct operations.
    stack(ax,x,y,13,8,'gold',layers=4)
    ax.add_patch(Circle((x+7,y+2.5),5.4,facecolor='white',edgecolor=C['gold'],lw=.9,zorder=7))
    txt(ax,x+7,y+3,str(number),8.2,'gold','bold','center',zorder=8)
    ax.add_patch(Arc((x+7,y+2.5),13.1,13.1,theta1=18,theta2=154,
                     color=C['gold'],lw=.9,zorder=7))
    arrow(ax,(x+1.2,y+5.7),(x+1.7,y+7.6),'gold',.75,rad=-.2,scale=5)

def dura_a(out):
    fig,ax=canvas(4.8);top(ax,'DURA: two learned-prior and consistency stages','restoration / A')
    txt(ax,3,48,'Received IQ',7,'ink','bold');waveform(ax,3,42,15,5,'blue',seed=5)
    txt(ax,3,34,'observation y',6.3,'muted')
    arrow(ax,(20,42),(26,49),'blue',.8,rad=-.12)
    arrow(ax,(20,40),(26,24),'blue',.8,rad=.12)
    txt(ax,28,54,'Coarse estimate',6.8,'ink','bold');waveform(ax,28,48,15,4,'teal',kind='restored')
    txt(ax,28,41,'initial state c',6.2,'muted')
    txt(ax,28,27,'Observation action',6.8,'ink','bold');response(ax,28,20,15,3.6)
    txt(ax,28,13,'estimated A and variance v',6.2,'muted')
    arrow(ax,(45,46),(45,24),'muted',.6,rad=-.12,ls='--',scale=4)
    arrow(ax,(44,48),(52,40),'teal',.8,rad=.12)
    path(ax,[(46,20),(50,20),(50,27),(101,27)],'gold',.65,ls='--',z=1)
    arrow(ax,(56,27),(56,30),'gold',.7,scale=5)
    arrow(ax,(79,27),(79,30),'gold',.7,scale=5)
    stage_glyph(ax,51,34,1);stage_glyph(ax,74,34,2)
    arrow(ax,(68,37),(73,37),'ink',1)
    txt(ax,58,23,'learned prior',6.2,'ink',ha='center')
    txt(ax,58,18,'+ 2 CG updates',6.0,'muted',ha='center')
    txt(ax,81,23,'learned prior',6.2,'ink',ha='center')
    txt(ax,81,18,'+ 2 CG updates',6.0,'muted',ha='center')
    arrow(ax,(91,37),(99,37),'ink',.9)
    waveform(ax,100,37,17,4,'teal',kind='restored');txt(ax,100,29,'restored IQ',6.4,'ink','bold')
    txt(ax,3,5,'Same observation, coarse estimate and action condition both stages.',6.2,'muted')
    save(fig,out,'DURA_A_unrolled')

def dura_b(out):
    fig,ax=canvas(5.0);top(ax,'DURA: an observation-constrained learned prior','restoration / B')
    txt(ax,3,53,'a',9,'ink','bold');txt(ax,8,53,'Two-stage restoration',7.0,'ink','bold')
    waveform(ax,6,42,16,4,'blue',seed=5);txt(ax,6,35,'received IQ',6.2,'muted')
    txt(ax,24,50,'coarse c + estimated A, v',6.1,'muted')
    arrow(ax,(23,42),(31,42),'blue',.9)
    stage_glyph(ax,32,39,1);stage_glyph(ax,58,39,2)
    arrow(ax,(50,42),(57,42),'gold',.9);arrow(ax,(76,42),(85,42),'gold',.9)
    waveform(ax,86,42,16,4,'teal',kind='restored');txt(ax,86,35,'restored IQ',6.2,'muted')
    txt(ax,40,30,'prior + 2 CG',6.0,'muted',ha='center')
    txt(ax,66,30,'prior + 2 CG',6.0,'muted',ha='center')
    path(ax,[(3,26),(117,26)],'grid',.65)
    txt(ax,3,22,'b',9,'ink','bold');txt(ax,8,22,'Inside each stage',7.0,'ink','bold')
    txt(ax,9,14,'observation residual',6.4,'blue','bold')
    txt(ax,9,8,'e = A* (y − A x)',6.4,'ink')
    arrow(ax,(35,12),(42,12),'muted',.8)
    txt(ax,44,14,'learned prior',6.4,'gold','bold')
    txt(ax,44,8,'p = c + τ D(·)',6.2,'ink')
    txt(ax,44,4,'D uses y, c, x, norm(e), log v',5.7,'muted')
    arrow(ax,(77,12),(84,12),'muted',.8)
    txt(ax,86,14,'soft consistency',6.4,'teal','bold')
    txt(ax,86,8,'fit y and stay near p',6.1,'ink')
    save(fig,out,'DURA_B_stage_mechanism')

def dura_c(out):
    fig,ax=canvas(4.6);top(ax,'DURA: waveform refinement under an estimated action','restoration / C')
    txt(ax,5,51,'OBSERVATION',6.3,'muted','bold');waveform(ax,5,44,20,6,'blue',seed=6)
    txt(ax,5,34,'y',7.2,'blue','bold')
    txt(ax,5,25,'coarse estimate c',6.5,'ink');waveform(ax,5,19,20,4,'teal',kind='restored')
    arrow(ax,(25,19),(36,28),'teal',.7,rad=-.12,ls='--',scale=5)
    arrow(ax,(26,44),(33,44),'blue',.8)
    for i,x in enumerate([38,70],1):
        # A strip of waveforms and layers gives each stage a physical visual form.
        ax.add_patch(Ellipse((x+8,38),27,30,fc=C['gold_light'],ec='none',alpha=.6,zorder=0))
        stack(ax,x,37,15,7,'gold',layers=4)
        waveform(ax,x-1,46,18,3,'gold',kind='restored',seed=i)
        txt(ax,x+8,28,f'STAGE {i}',6.6,'gold','bold','center')
        txt(ax,x+8,22,'learned prior',6.2,'ink',ha='center')
        txt(ax,x+8,17,'2 CG updates',6.1,'muted',ha='center')
    arrow(ax,(56,43),(69,43),'ink',.95);arrow(ax,(89,43),(98,43),'ink',.95)
    waveform(ax,99,43,16,5,'teal',kind='restored');txt(ax,100,34,'restored IQ',6.5,'ink','bold')
    path(ax,[(27,13),(93,13)],'violet',.75,ls='--')
    for x in [45,77]:arrow(ax,(x,13),(x,17),'violet',.75,scale=4)
    txt(ax,61,7,'shared observation action A and variance v',6.2,'muted',ha='center')
    save(fig,out,'DURA_C_waveform_refinement')

def dura_d(out):
    fig,ax=canvas(5.35);top(ax,'DURA architecture and stage computation','restoration / D')
    txt(ax,3,55,'a',9,'ink','bold');txt(ax,8,55,'Shared estimation and two unrolled stages',7.0,'ink','bold')
    waveform(ax,4,44,12,3.5,'blue',seed=5);txt(ax,4,38,'received IQ  y',6.1,'muted')
    arrow(ax,(17,44),(23,49),'blue',.7,rad=-.1)
    arrow(ax,(17,42),(23,33),'blue',.7,rad=.1)
    stack(ax,24,47,11,4,'teal',layers=3);txt(ax,24,51.2,'coarse estimate',6.1,'ink','bold')
    txt(ax,24,42,'frequency + time',5.8,'muted')
    response(ax,24,34,12,2.2);txt(ax,24,27,'estimated A, v',5.9,'muted')
    arrow(ax,(37,48),(47,43),'teal',.7);arrow(ax,(36,46),(36,36),'muted',.6,ls='--',scale=4)
    stage_glyph(ax,48,39,1);stage_glyph(ax,71,39,2)
    arrow(ax,(66,42),(70,42),'gold',.85);arrow(ax,(88,42),(97,42),'gold',.85)
    waveform(ax,99,42,16,3.2,'teal',kind='restored');txt(ax,99,35,'restored IQ',6.1,'muted')
    path(ax,[(37,33),(43,33),(43,30),(88,30)],'violet',.65,ls='--')
    for x in [55,78]:arrow(ax,(x,30),(x,35),'violet',.65,scale=4)
    txt(ax,57,24,'observation, coarse estimate and action are shared',6.0,'muted',ha='center')
    path(ax,[(3,20),(117,20)],'grid',.65)
    txt(ax,3,16,'b',9,'ink','bold');txt(ax,8,16,'Stage k',6.8,'ink','bold')
    txt(ax,8,10,'e = A*(y − A x)',6.3,'blue')
    arrow(ax,(31,10),(37,10),'muted',.7)
    txt(ax,39,12,'learned prior',6.2,'gold','bold')
    txt(ax,39,7,'p = c + τD(y, c, x, norm(e), log v)',5.8,'ink')
    arrow(ax,(80,10),(85,10),'muted',.7)
    txt(ax,87,12,'soft consistency',6.1,'teal','bold')
    txt(ax,87,7,'2 CG updates',5.9,'muted')
    save(fig,out,'DURA_D_network_detail')

def earth(ax):
    verts=[(-1,4),(-1,19),(12,24),(25,24),(37,20),(50,17),(62,14),(68,9),
           (68,0),(-1,0),(-1,4)]
    codes=[Path.MOVETO,Path.LINETO,Path.CURVE4,Path.CURVE4,Path.CURVE4,
           Path.CURVE4,Path.CURVE4,Path.CURVE4,Path.LINETO,Path.LINETO,Path.CLOSEPOLY]
    ax.add_patch(PathPatch(Path(verts,codes),fc='#e8f1ef',ec='none',zorder=0))
    path(ax,[(0,18),(9,21),(18,22),(27,21.5),(37,19),(48,16),(59,12),(68,9)],'blue',.6,alpha=.35,z=1)
    # Abstract terrain contours, not a geographic claim.
    for off in [0,2.5,5.0]:
        path(ax,[(0,9+off),(6,10+off),(11,8+off),(17,10+off),(23,8.5+off),
                 (30,9+off),(37,7+off)],'teal',.45,alpha=.3,z=1)

def transmitter(ax,x,y,s=1):
    path(ax,[(x,y),(x,y+10*s)],'ink',1)
    path(ax,[(x-3*s,y),(x+3*s,y)],'ink',.9)
    path(ax,[(x-2.2*s,y),(x,y+6*s),(x+2.2*s,y)],'muted',.7)
    ax.add_patch(Circle((x,y+10*s),.85*s,fc=C['gold'],ec='white',lw=.4,zorder=7))
    for r in [2.4,4.0,5.7]:
        ax.add_patch(Arc((x,y+10*s),r*2,r*2,theta1=15,theta2=80,color=C['gold'],lw=.6,alpha=.8))

def satellite(ax,x,y,s=1):
    # Hand-drawn vector satellite: bus, solar cells, antenna and structural lines.
    bus=Polygon([(x-2.8*s,y-2*s),(x+2.3*s,y-2*s),(x+3*s,y+1*s),
                 (x-2*s,y+2*s)],fc='#fff9e9',ec=C['ink'],lw=.75,zorder=6)
    ax.add_patch(bus)
    ax.add_patch(Polygon([(x+2.3*s,y-2*s),(x+3*s,y+1*s),(x+4.1*s,y+.45*s),
                          (x+3.5*s,y-2.55*s)],fc='#e2d2b5',ec=C['ink'],lw=.5,zorder=5))
    ax.add_patch(Polygon([(x-2*s,y+2*s),(x+3*s,y+1*s),(x+4.1*s,y+1.7*s),
                          (x-.8*s,y+2.8*s)],fc='#f4ebd9',ec=C['ink'],lw=.45,zorder=5))
    for side in [-1,1]:
        bx=x+(3.0 if side>0 else -12.5)*s
        panel=Polygon([(bx,y-.7*s),(bx+9.2*s,y-.7*s),(bx+9.2*s,y+1.7*s),(bx,y+1.7*s)],
                      fc=C['blue_light'],ec=C['blue'],lw=.65,zorder=5)
        ax.add_patch(panel)
        for k in range(1,5):path(ax,[(bx+k*1.84*s,y-.7*s),(bx+k*1.84*s,y+1.7*s)],'blue',.35,z=6)
        path(ax,[(bx,y+.5*s),(bx+9.2*s,y+.5*s)],'blue',.35,z=6)
    path(ax,[(x-.8*s,y-2*s),(x-1.4*s,y-4*s)],'ink',.65)
    ax.add_patch(Arc((x-1.4*s,y-4*s),3.8*s,2.0*s,theta1=185,theta2=355,color=C['ink'],lw=.7,zorder=7))
    path(ax,[(x-1.4*s,y-4*s),(x-1.4*s,y-2.7*s)],'ink',.55)
    ax.add_patch(Circle((x+.2*s,y+.2*s),.65*s,fc=C['gold'],ec=C['ink'],lw=.35,zorder=7))
    path(ax,[(x+1.2*s,y+.4*s),(x+2.0*s,y+.4*s)],'ink',.4,z=7)
    path(ax,[(x-1.5*s,y-.9*s),(x-.3*s,y-.9*s)],'gold',.55,z=7)

def processing_chip(ax,x,y,color,label,small):
    w=19;h=9
    for layer in range(3):
        yy=y+layer*.5
        poly=Polygon([(x,yy),(x+w,yy),(x+w+2,yy+2),(x+2,yy+2)],closed=True,
                     fc=C[color],ec=C[color],lw=.45,alpha=.11+layer*.045,zorder=2+layer)
        ax.add_patch(poly)
    for k in range(3):
        path(ax,[(x+2+k*5,y+2),(x+3+k*5,y+3)],color,.55,alpha=.8,z=6)
    txt(ax,x+w/2,y+6,label,7.1,color,'bold','center')
    txt(ax,x+w/2,y-3,small,6.0,'muted',ha='center')

def system_a(out):
    fig,ax=canvas(4.8);top(ax,'Passive RF fingerprinting from a spaceborne receiver','system / A')
    earth(ax);transmitter(ax,10,17,.85);satellite(ax,42,47,1.1)
    # Propagation beam is a translucent wedge, bounded by the ground emitter and the receiver.
    beam=Polygon([(10,25),(36,46),(43,44),(16,23)],fc=C['gold'],ec='none',alpha=.09,zorder=1)
    ax.add_patch(beam)
    for dx in [0,1.8,3.5]:
        path(ax,[(11+dx,26),(25+dx,36),(38+dx,45)],'gold',.5,alpha=.55,ls='--',z=2)
    txt(ax,3,9,'terrestrial emitter',6.3,'ink','bold')
    txt(ax,35,54,'satellite receiver',6.6,'ink','bold')
    txt(ax,19,30,'space–ground\npropagation',6.1,'muted',ha='center')
    path(ax,[(28,56),(38,58),(49,57),(58,53)],'blue',.55,alpha=.55,ls='--')
    arrow(ax,(54,47),(66,47),'blue',.8)
    txt(ax,67,53,'received IQ',6.7,'ink','bold');waveform(ax,67,47,15,3,'blue',seed=6)
    processing_chip(ax,88,42,'rose','PAMF','frozen identifier')
    arrow(ax,(83,47),(87,47),'rose',.75)
    processing_chip(ax,67,25,'gold','DURA','optional restoration')
    arrow(ax,(76,44),(76,34),'gold',.7)
    arrow(ax,(87,28),(97,28),'gold',.8);arrow(ax,(98,28),(98,39),'gold',.75)
    arrow(ax,(110,47),(117,47),'ink',.8);txt(ax,116,40,'identity',6.5,'ink',ha='right')
    txt(ax,69,12,'The same PAMF classifier is used with or without DURA.',5.8,'muted')
    txt(ax,69,6,'Concept: ground IQ + simulated channel in evaluation.',5.8,'muted')
    save(fig,out,'System_A_orbital_context')

def system_b(out):
    fig,ax=canvas(5.0);top(ax,'Source-only learning and passive inference','system / B')
    txt(ax,3,53,'a',9,'ink','bold');txt(ax,8,53,'Source development',7.2,'ink','bold')
    txt(ax,8,46,'labeled source IQ',6.4,'blue','bold');waveform(ax,8,41,18,3.5,'blue',seed=1)
    txt(ax,8,35,'unlabeled source IQ',6.4,'teal','bold');waveform(ax,8,30,18,3.5,'teal',seed=2)
    ribbon(ax,(28,41),(46,42),2,'blue',.2);ribbon(ax,(28,30),(46,38),2,'teal',.2)
    processing_chip(ax,48,38,'rose','PAMF','Tri-SIG-TIME SSL')
    txt(ax,49,26,'source IQ pairs (L or L+U)\nno TX labels',5.9,'muted');processing_chip(ax,78,32,'gold','DURA','independent; optional')
    arrow(ax,(28,30),(77,34),'gold',.65,rad=.15,ls='--')
    path(ax,[(27,41),(31,41),(31,23.7),(76,23.7),(76,34)],'gold',.6)
    arrow(ax,(76,34),(78,34),'gold',.65,scale=4)
    txt(ax,101,43,'freeze',6.3,'muted','bold');path(ax,[(104,39),(104,30)],'muted',.7,ls='--')
    path(ax,[(3,23),(117,23)],'grid',.65)
    txt(ax,3,19,'b',9,'ink','bold');txt(ax,8,19,'Passive inference',7.1,'ink','bold')
    waveform(ax,8,10,16,3,'blue',seed=4);txt(ax,8,5,'unseen receiver IQ',6.0,'muted')
    arrow(ax,(25,10),(42,10),'blue',.75);processing_chip(ax,43,7,'gold','DURA','optional')
    arrow(ax,(64,10),(79,10),'gold',.75);processing_chip(ax,80,7,'rose','PAMF','frozen')
    path(ax,[(26,12),(26,16),(78,16),(78,12)],'blue',.7)
    arrow(ax,(78,12),(80,12),'blue',.7);txt(ax,51,18,'direct path',5.9,'muted',ha='center')
    arrow(ax,(101,10),(107,10),'ink',.8);txt(ax,109,10,'identity',6.2,'ink')
    save(fig,out,'System_B_training_and_inference')

def system_c(out):
    fig,ax=canvas(4.7);top(ax,'Spaceborne reception with two inference paths','system / C')
    earth(ax);transmitter(ax,12,18,.8);satellite(ax,39,46,1.0)
    ax.add_patch(Polygon([(12,26),(35,44),(41,44),(16,25)],fc=C['blue'],ec='none',alpha=.08,zorder=1))
    path(ax,[(13,27),(25,37),(37,45)],'gold',.65,alpha=.65,ls='--')
    txt(ax,4,10,'emitter',6.5,'ink','bold');txt(ax,31,53,'spaceborne receiver',6.6,'ink','bold')
    arrow(ax,(50,46),(59,46),'blue',.8)
    waveform(ax,60,46,17,3.5,'blue',seed=4);txt(ax,60,39,'received IQ',6.3,'muted')
    # Direct path is upper; restored path is lower, both terminate in the same F.
    path(ax,[(78,47),(84,47),(84,52),(96,52)],'blue',.7)
    arrow(ax,(95,52),(97,52),'blue',.7);txt(ax,87,55,'direct',6.1,'blue','bold')
    processing_chip(ax,65,23,'gold','DURA','optional')
    arrow(ax,(69,42),(69,32),'gold',.8)
    path(ax,[(86,26),(94,26),(94,46)],'gold',.75);arrow(ax,(94,46),(97,46),'gold',.75)
    processing_chip(ax,97,46,'rose','PAMF','frozen classifier')
    arrow(ax,(107,44),(107,38),'rose',.75)
    txt(ax,99,34,'TX identity',6.4,'ink','bold')
    txt(ax,61,7,'Source-trained models are frozen; no target-domain updating.',6.1,'muted')
    save(fig,out,'System_C_inference_paths')

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=FilePath,required=True)
    a=parser.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    functions=[network_a,network_b,network_c,network_d,dura_a,dura_b,dura_c,dura_d,system_a,system_b,system_c]
    for fn in functions:
        fn(a.output)
        print('Rendered',fn.__name__,flush=True)
    names=['PAMF_A_multibranch','PAMF_B_feature_fan','PAMF_C_signature_detail','PAMF_D_architecture_detail',
           'DURA_A_unrolled','DURA_B_stage_mechanism','DURA_C_waveform_refinement','DURA_D_network_detail',
           'System_A_orbital_context','System_B_training_and_inference','System_C_inference_paths']
    manifest={'backend':'Python / Matplotlib','editable_text':True,'raster_embedded_in_svg':False,
              'figure_width_mm':182.88,'palette':'shared muted blue, teal, rose and amber',
              'visuals':'schematic waveform illustrations, not data',
              'formats':['svg','pdf','png'],'figures':names}
    (a.output/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    (a.output/'layout_diagnostics.json').write_text(json.dumps(QA,ensure_ascii=False,indent=2),encoding='utf-8')
    if any(q['text_outside_canvas'] for q in QA):
        raise ValueError('Figure text extends outside canvas: '+json.dumps(QA,ensure_ascii=False))

if __name__=='__main__':main()
