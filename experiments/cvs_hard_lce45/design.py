"""One fixed L-only hard-CE ablation; all IQ/U/forward budgets unchanged."""
from pathlib import Path
from copy import deepcopy
from experiments.cvs_noise_shape_factorial43 import design as old
ROOT=Path(__file__).resolve().parents[2]
PROJECT=old.PROJECT;SOURCE=old.SOURCE;FULL_FP32_POLICY=old.FULL_FP32_POLICY
CLASSES=old.CLASSES;SCENES=old.SCENES;VIEWS=old.VIEWS;FORMAL_VIEWS=old.FORMAL_VIEWS
TRUTH=old.TRUTH;PANEL=old.PANEL;SOURCE_CACHE=old.SOURCE_CACHE
RUN='20261018-phase1-FcrossRX-hardLCE-e30-s2026092701-m2-r01'
RELEASE='cvs_FcrossRX_hardLCE_E30_20261018_r01';OWNER='codex/root/hardLCE45'
BASE=Path(PROJECT)/'runs'/RUN;REFERENCE=old.BASE
RX=FIT=(1,3,4,6,8);SEED=2026092701;STEPS=222;UPDATES=6660
ARMS=('H00','H01');read,write=old.read,old.write

def rows():return [dict(arm=a,row_id=a+'-s'+str(SEED),model_seed=SEED) for a in ARMS]
def reference_config(c):
 arm='B01' if c['As'] else 'B00'
 return old.config(next(r for r in old.rows() if r['arm']==arm),c['segment'])
def config(row,segment=30):
 if row not in rows() or segment not in (20,30):raise ValueError('Only two scratch20+own10 rows')
 arm=row['arm'];sat=arm=='H01';base=next(r for r in old.rows() if r['arm']==('B01' if sat else 'B00'))
 c=deepcopy(old.config(base,segment));tail=segment==30;parent=BASE/row['row_id']/'stage20'
 c.update(schema='FcrossRX_hard_lce45_v1',run_id=RUN,**row,launch_owner=OWNER,output_root=str(BASE/row['row_id']/('source' if tail else 'stage20')),parent_source=str(parent) if tail else None,checkpoint_sources=[str(parent/'final_ssdg.pth')] if tail else [],new_Ag=False,As=sat,Ag_mode='NONE',L_label_smoothing=0.,native_and_U_label_smoothing=.01,loss_only_ablation=True,training_recipe='Exact P43 noAg controls:6x128 L NN/step;.75ordinary+.25As_or_raw plus .25raw;replace L CE.01 by CE0 only;unchanged predictive RDrop1/SupCon/oldU;own20+10')
 c['training_mechanism'].update(method='L_only_CE0_ablation',L_forwards_per_sample=6,total_L_loss_scale=1.25,L_label_smoothing=0.,native_and_U_label_smoothing=.01,new_U_augmentation=False,extra_U_NN=0)
 c['baseline_reference']='Immutable P43 B00/B01 corresponding FINAL30;source-only evidence motivated fixed hardLCE,not a new IQAg;original U remains untouched'
 c['benchmark_context']='Exposed benchmark context retained;P40 CE0/CE01 separates calibration from identity derivatives,does not establish training gain or rescue rejected noise-gradient hypothesis'
 return c

def validate(c):
 row=next((r for r in rows() if r['row_id']==c.get('row_id')),None)
 if row is None or c!=config(row,c.get('segment')):raise ValueError('Fixed L-only/allsource/E30/defaulthead/U/6L contract differs')
 return c

def make_args(c,device='cuda:0'):
 validate(c);a=old.make_args(reference_config(c),device)
 a.run_id=RUN;a.candidate_id=c['row_id'];a.output_dir=c['output_root'];a.baseline_ckpt='' if c['segment']==20 else c['parent_source']+'/final_ssdg.pth'
 if float(a.label_smoothing)!=.01:raise ValueError('Native/U smoothing must remain original .01')
 return a
