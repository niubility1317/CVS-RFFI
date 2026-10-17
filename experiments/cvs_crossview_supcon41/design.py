"""Fixed existing SupCon budget; compare ordinary versus duplicate versus real satellite coverage."""
from copy import deepcopy
from pathlib import Path
from experiments.cvs_clock_factorial20 import design as old
ROOT=Path(__file__).resolve().parents[2];PROJECT=old.PROJECT;SOURCE=old.SOURCE
RUN='20261018-phase1-FcrossRX-crossviewSupCon-e30-s2026092701-m3-r01';RELEASE='cvs_FcrossRX_crossviewSupCon_E30_20261018_r01';OWNER='codex/root/crossviewSupCon41';BASE=Path(PROJECT)/'runs'/RUN
read,write=old.read,old.write
SEED=2026092701;STEPS=222;UPDATES=6660;FIT=RX=(1,3,4,6,8)
FULL_FP32_POLICY=old.FULL_FP32_POLICY;CLASSES=old.CLASSES;SCENES=old.SCENES;VIEWS=old.VIEWS;TRAIN_SCENES=old.reference_config()['channel']['train_scenes']
FORMAL_VIEWS=Path(PROJECT)/'runs/20261002-phase1-cvs-residual-sixscene-manysig-m8-r01/received_views'
TRUTH=old.TRUTH
# Checked against the existing clean+6 received-IQ manifest before launch.
SMOKE_SOURCE=Path(PROJECT)/'runs/20261018-phase1-FcrossRX-cleanurban-U-e30-s2026092701-m3-r01/B0-s2026092701/source'
ARMS=('ORD128','DUP192','SAT192')
PANEL=Path(PROJECT)/'runs/20261017-diagnostic-FcrossRX-fullsource-residual-s2026092701-m3-r01/diagnostic'
SOURCE_CACHE=Path(PROJECT)/'runs/20261011-phase1-stage2-geometry-source-manysig-m3-r01/s2026092701'
def rows():return [dict(arm=a,row_id=a+'-s'+str(SEED),model_seed=SEED) for a in ARMS]
def config(row,segment=30):
 if row not in rows() or segment not in (20,30):raise ValueError('Only three independent scratch20+own10 rows')
 c=deepcopy(old.config(next(r for r in old.rows() if r['arm']=='B01')));is_tail=segment==30
 c.update(schema='FcrossRX_crossview_supcon41_v1',run_id=RUN,**row,launch_owner=OWNER,output_root=(BASE/row['row_id']/('source' if is_tail else 'stage20')).as_posix(),baseline_reuse=False,epochs=10 if is_tail else 20,label_epochs=0 if is_tail else 13,pseudo_epochs=10 if is_tail else 7,updates=2220 if is_tail else 4440,segment=segment,epoch_offset=20 if is_tail else 0,step_offset=4440 if is_tail else 0,cumulative_epochs=30 if is_tail else 20,cumulative_updates=6660 if is_tail else 4440,parent_source=(BASE/row['row_id']/'stage20').as_posix() if is_tail else None,checkpoint_sources=[] if not is_tail else [(BASE/row['row_id']/'stage20/final_ssdg.pth').as_posix()],model_initialization='own_stage20' if is_tail else 'scratch',training_recipe='Original complete cleanretain F+B01:13L17U/4x128/rawnativeperpacket.75ordinary.25sat/cos20/resetcos10;'+row['arm'],native_scratch_only=not is_tail,new_Ag=False,As=True)
 c['U_transfer']=dict(mode='ORIGINAL_ONLY',extra_student_packets=0,extra_teacher_packets=0,new_loss=False,U_TX_hidden=True)
 c['crossview']=dict(mode=row['arm'],ordinary_anchors=128,additional_anchors=0 if row['arm']=='ORD128' else 64,additional_view='none' if row['arm']=='ORD128' else ('same ordinary maps duplicated at satellite physical IDs' if row['arm']=='DUP192' else 'actual full satellite maps at same physical IDs'),positive='sameTX differentphysicalRX',same_physical_positive=False,default_head_unchanged=True,new_losses=0,new_forward_packets=0,temperature=.2,max_coefficient=.1,schedule='original warm5/ramp10;tail original cumulativeE21..30',normalization='descriptor moments detached across actual anchor set;L2 eps1e-4;mean bothdepths and two corresponding RDrop forwards')
 c['training_mechanism']['new_U_augmentation']=False;c['training_mechanism']['no_new_U_consistency']=True;c['training_mechanism']['expected_L_forward_packets']=c['updates']*512;c['training_mechanism']['original_L_identity_forwards']=c['updates']*128;c['training_mechanism']['extra_L_identity_forwards']=c['updates']*384
 c['baseline_reference']='All3 scratchE0 plus own20 only;original U only,no extraU student/teacher;4L NN matched;no historical controlweights'
 c['benchmark_context']='Human selected exposed-benchmark full training organization;source-only code evidence:old SupCon excludes all satellite maps;old weights never initialize new training;fixedfinalthenautomaticall7truthlast'
 return c
def validate(c):
 row=next((r for r in rows() if r['row_id']==c.get('row_id')),None)
 if row is None or c!=config(row,c.get('segment')):raise ValueError('Fixed all-source/E30/defaulthead/oldU/fullrecipe contract differs')
 return c
def make_args(c,device='cuda:0'):
 validate(c);base=old.config(next(r for r in old.rows() if r['arm']=='B01'));a=old.make_args(base,device)
 a.run_id=RUN;a.candidate_id=c['row_id'];a.output_dir=c['output_root'];a.epochs=c['epochs'];a.label_epochs=c['label_epochs'];a.pseudo_epochs=c['pseudo_epochs'];a.from_scratch=c['segment']==20;a.a1_scratch_only=a.from_scratch;a.baseline_ckpt='' if a.from_scratch else c['parent_source']+'/final_ssdg.pth';a.teacher_ckpt='';return a
