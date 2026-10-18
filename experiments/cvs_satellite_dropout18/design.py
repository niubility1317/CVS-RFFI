"""Satellite subnetwork noise at fixed full practical and matched E30 costs."""
import json
from copy import deepcopy
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
RUN='20261018-phase1-FcrossRX-satelliteDropout-e30-s2026092701-m2-r01';RELEASE='cvs_F_satellite_dropout_E30_20261018_r01';OWNER='codex/root/satellite-dropout18';LOCAL='local_artifacts/F_satellite_dropout_E30_20261018'
REFERENCE_RUN='20261018-phase1-FcrossRX-impulse-factorial-e30-s2026092701-m6-r01'
ARMS=('AS_CE','AS_DET');SEED=2026092701;RX=FIT=(1,3,4,6,8);STEPS=222;UPDATES=6660

def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
_runtime=ROOT/'runtime.json'
PROJECT=read(_runtime)['project'] if _runtime.is_file() else 'PROJECT'
def resolved(v):
 if isinstance(v,dict):return {k:resolved(x) for k,x in v.items()}
 if isinstance(v,list):return [resolved(x) for x in v]
 return v.replace('PROJECT',PROJECT) if isinstance(v,str) else v
_REFERENCE=resolved(read(Path(__file__).with_name('reference_config.json')))
globals().update(_REFERENCE['constants'])
SOURCE=Path(SOURCE);FORMAL_VIEWS=Path(FORMAL_VIEWS);TRUTH=Path(TRUTH);PANEL=Path(PANEL);SOURCE_CACHE=Path(SOURCE_CACHE);BASE=Path(PROJECT)/'runs'/RUN

def rows():return [dict(arm=a,row_id=a+'-s'+str(SEED),model_seed=SEED) for a in ARMS]
def config(row,segment=30):
 if row not in rows() or segment not in (20,30):raise ValueError('Only two fixed newrows/own20+10')
 c=deepcopy(_REFERENCE['configs'][str(segment)]);tail=segment==30;out=BASE/row['row_id']/('source' if tail else 'stage20');parent=BASE/row['row_id']/'stage20'
 c.update(schema='FcrossRX_satellite_dropout18_v1',run_id=RUN,**row,launch_owner=OWNER,output_root=str(out),parent_source=str(parent) if tail else None,checkpoint_sources=[str(parent/'final_ssdg.pth')] if tail else [],new_Ag=False,As=True,Ag_mode='NONE',training_recipe='OriginalP43 .5ordinaryRD+.5mixedRDwithsatelliteKLoff+.25rawRD;onlyAS_DET bypasses existingdropout on actual64satellite rows;6Lforwards/oldU',satellite_mode=row['arm'],goal='Maximize fixedpractical identification;stageclean drop<=0.1absolutepp vs matchedNoAs;finalrepeatablecleanpositive;no mandatoryAg',budget_authorization='Latest userE30 permodel;own20+10 total6660updates;2newmodels only',cost_claim='6Lforwards/1.25Lscale same as reusedP43B00/B01;new60modelepochs;historicalcontrols60modelepochs separate',baseline_reference='Reused originalP43 B00/B01 seed01 exactfullsource/factory/ordinary/U/6L budget;AS_CE attributioncontrol,not revivalofoldheldRXAsKLrecipe',source_selection='Fixedfinal30;sourcefullVclean>=NoAs-.1pp and sourcepanelmidlow usefulincrement;no perpacketidentity orconfidence filter',baseline_reuse=False,benchmark_context='Human exposed benchmark background;new mechanism from training perturbationcomposition,not testselection;no targetfeedback')
 c['training_mechanism'].update(expected_L_forward_packets=c['updates']*768,extra_L_identity_forwards=c['updates']*640)
 c['satellite_dropout']=dict(mode=row['arm'],mask='Actual alternating64 physicalsatellite rows from existingstep parity',AS_CE='Originalstochasticdropout;only satelliteKLoff',AS_DET='Same CE,existingnn.Dropout output bypass onlysatellite rows;still draw nativefullmask to match RNG',ordinary_raw_U='Originaldropout/RDrop unchanged',new_parameters=0,new_deployment_modules=0,extra_forwards=0,all_labeled_satellite_supervised=True)
 return c
def validate(c):
 row=next((r for r in rows() if r['row_id']==c.get('row_id')),None)
 if row is None or c!=config(row,c.get('segment')):raise ValueError('Fixed satellite dropout matrix/data/epoch differs')
 return c
def make_args(c,device='cuda:0'):
 validate(c)
 from experiments.cvs_noise_shape_factorial43 import design as b
 a=b.make_args(b.config(next(r for r in b.rows() if r['arm']=='B01'),c['segment']),device);a.run_id=RUN;a.candidate_id=c['row_id'];a.output_dir=c['output_root'];a.baseline_ckpt='' if c['segment']==20 else c['parent_source']+'/final_ssdg.pth';return a
