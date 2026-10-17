"""Prospective 3x2 IQ noise-shape/actualAs factorial;all-source own20+10."""
from pathlib import Path
from copy import deepcopy
import json
from experiments.cvs_crossview_supcon41 import design as b
ROOT=Path(__file__).resolve().parents[2];PROJECT=b.PROJECT;SOURCE=b.SOURCE;FULL_FP32_POLICY=b.FULL_FP32_POLICY;CLASSES=b.CLASSES;SCENES=b.SCENES;VIEWS=b.VIEWS;FORMAL_VIEWS=b.FORMAL_VIEWS;TRUTH=b.TRUTH;PANEL=b.PANEL;SOURCE_CACHE=b.SOURCE_CACHE
RUN='20261018-phase1-FcrossRX-impulse-factorial-e30-s2026092701-m6-r01';RELEASE='cvs_FcrossRX_impulse_factorial_E30_20261018_r01';OWNER='codex/root/noise-shape-factorial43';BASE=Path(PROJECT)/'runs'/RUN
RX=FIT=(1,3,4,6,8);SEED=2026092701;STEPS=222;UPDATES=6660
ARMS=('B00','B10_WHITE','B10_BG','B01','B11_WHITE','B11_BG');read,write=b.read,b.write
from .noise import NOISE
def rows():return [dict(arm=a,row_id=a+'-s'+str(SEED),model_seed=SEED) for a in ARMS]
def config(row,segment=30):
 if row not in rows() or segment not in (20,30):raise ValueError('Only fixed six rows and own20+10')
 c=deepcopy(b.config(b.rows()[0],segment));tail=segment==30;arm=row['arm'];mode='BG' if arm.endswith('_BG') else 'WHITE' if arm.endswith('_WHITE') else 'NONE';sat=arm.startswith('B01') or arm.startswith('B11')
 out=BASE/row['row_id']/('source' if tail else 'stage20');parent=BASE/row['row_id']/'stage20'
 c.update(schema='FcrossRX_noise_shape_factorial43_v1',run_id=RUN,**row,launch_owner=OWNER,output_root=str(out),parent_source=str(parent) if tail else None,checkpoint_sources=[str(parent/'final_ssdg.pth')] if tail else [],new_Ag=mode!='NONE',As=sat,Ag_mode=mode,noise=deepcopy(NOISE),L_NN_per_step=768,total_L_loss_scale=1.25,training_recipe='Sharedfullcleanretain .75ordinary+.25As_or_raw plus .25Ag_or_raw;6x128sameIQ-RDrop;originalSupCononordinary128;originalUonly;own20+10')
 c['crossview'].update(mode='ORD128',additional_anchors=0,additional_view='none')
 c['training_mechanism'].update(expected_L_forward_packets=c['updates']*768,extra_L_identity_forwards=c['updates']*640)
 c['baseline_reference']='Allsix own scratch20+own10;6LNN matched,common ordinary .75 and fullAs .25 not diluted;Ag control raw .25;original U unchanged;loss total1.25 in allrows'
 c['benchmark_context']='Human-exposed benchmark context retained;no target-based recipe selection;new candidate fixed from nonGaussian observation hypothesis,not observed WiSig EMI claim'
 return c
def validate(c):
 row=next((r for r in rows() if r['row_id']==c.get('row_id')),None)
 if row is None or c!=config(row,c.get('segment')):raise ValueError('Fixed sixrow/E30/allsource/defaulthead/oldU/noiseshape contract differs')
 return c
def make_args(c,device='cuda:0'):
 validate(c);a=b.make_args(b.config(b.rows()[0],c['segment']),device);a.run_id=RUN;a.candidate_id=c['row_id'];a.output_dir=c['output_root'];a.baseline_ckpt='' if c['segment']==20 else c['parent_source']+'/final_ssdg.pth';return a
