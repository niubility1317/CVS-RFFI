"""Four counterfactuals with identical raw-secondary input and scratch E30 cost."""
from copy import deepcopy
from pathlib import Path
from experiments.cvs_fullsource_concat30_25 import design as ref
ROOT=Path(__file__).resolve().parents[2]
PROJECT,SOURCE,FULL_FP32_POLICY,read,write,TRUTH,CLASSES,SMOKE_SOURCE,FORMAL_VIEWS,SCENES,VIEWS,TRAIN_SCENES,SEED,STEPS,UPDATES,RX=(getattr(ref,k) for k in ('PROJECT','SOURCE','FULL_FP32_POLICY','read','write','TRUTH','CLASSES','SMOKE_SOURCE','FORMAL_VIEWS','SCENES','VIEWS','TRAIN_SCENES','SEED','STEPS','UPDATES','RX'))
RUN='20261017-phase1-FcrossRX-cycleorder-factorial-e30-s2026092701-m4-r01';RELEASE='cvs_FcrossRX_cycleorder_factorial_E30_20261017_r01';OWNER='codex/root/FcrossRX-cycleorder-factorial-E30-20261017';BASE=Path(PROJECT)/'runs'/RUN
parent_config=ref.parent_config
def rows():return [dict(arm=a,row_id=a+'-s'+str(SEED),model_seed=SEED) for a in ('B00','B10','B01','B11')]
def config(row):
 if row not in rows():raise ValueError('Only fixed four full-source rows allowed')
 ag=row['arm'] in ('B10','B11');sat=row['arm'] in ('B01','B11');prior=next(r for r in ref.rows() if r['arm']==('B01' if sat else 'B00'));c=deepcopy(ref.config(prior))
 c.update(schema='FcrossRX_fixed_cycleorder_rawsecondary_E30_v1',run_id=RUN,**row,launch_owner=OWNER,output_root=(BASE/row['row_id']/'source').as_posix(),Ag=ag,As=sat,training_recipe='common ordinary(x) primary;raw/Ag(raw)/As(raw)/As(Ag(raw)) secondary;.5 each normalized original RDrop;SupCon primary;20L10U/fullsource/scratch30')
 c['training_mechanism'].update(clean_repeat_control=False,primary='unchanged ordinary(x)',secondary='As(Ag(raw))' if ag and sat else 'Ag(raw)' if ag else 'As(raw)' if sat else 'raw',paired_clean_raw_control=True)
 c['channel']['input']='raw normalized physical L'+(' after fixed Ag' if ag else '');c['cycle_order']=dict(enabled=ag,operator='immutable P30 phase-transported S4',window=[80,160],period=20,private_seed=202610173001,view_namespace='trainL/epoch{epoch}/step{step}/secondary',label_selection=False,new_U_Ag=False,As_repeats=1 if sat else 0,extra_network_forwards=0)
 c['source_basis']=dict(run='20261017-diagnostic-FcrossRX-cycleorder-placement-sourceV-e30-s2026092701-m2-r01',decision='PASS_SOURCE_PLACEMENT_ONLY;not proof of Ag benefit',post_order='FAIL retained;never used',fixed_threshold='each two-draw urban difference>=-.5pp',ordinary_secondary_reference=ref.RUN,reference_caveat='New four raw-secondary rows required for causal matching. Historical ordinary-secondary E30 is direct external reference, not a same-input counterfactual. Historical raw-As E20 negative not reclassified; raw rows are required controls for new fixed Ag.')
 c['source_qualification']=dict(rule='B10 fullV>=B00+.5pp;B11 fullV>=B00 and each urban draw>=B01+1pp;all-source,noholdout;source-only extension decision',automatic_seed_extension=False,all_four_rows_formal=True)
 return c
def validate(c):
 r=next((r for r in rows() if r['row_id']==c.get('row_id')),None)
 if r is None or c!=config(r):raise ValueError('Fixed full-source four-row E30 config required')
 return c
def make_args(c,device='cuda:0'):
 validate(c);prior=next(r for r in ref.rows() if r['arm']==('B01' if c['As'] else 'B00'));a=ref.make_args(ref.config(prior),device);a.run_id=RUN;a.candidate_id=c['row_id'];a.output_dir=c['output_root'];return a
