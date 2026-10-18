"""One fixed nominal As pool: random CE control versus quality-stratified risk."""
from copy import deepcopy
from pathlib import Path
from experiments.cvs_noise_shape_factorial43 import design as old
ROOT=Path(__file__).resolve().parents[2]
PROJECT,SOURCE=old.PROJECT,old.SOURCE
RUN='20261018-phase1-FcrossRX-qualityStrat-e30-s2026092701-m2-r01'
RELEASE='cvs_FcrossRX_quality_strat_E30_20261018_r01'
OWNER='codex/root/quality-risk70'
BASE=Path(PROJECT)/'runs'/RUN
REFERENCE_RUN=old.RUN
SEED,STEPS,UPDATES=2026092701,222,2220
RX=FIT=(1,3,4,6,8)
ARMS=('M1','M2')
LOCAL_NN,SMOKE_NN=0,4
FULL_FP32_POLICY,CLASSES,SCENES,VIEWS=old.FULL_FP32_POLICY,old.CLASSES,old.SCENES,old.VIEWS
FORMAL_VIEWS,TRUTH,PANEL,SOURCE_CACHE=old.FORMAL_VIEWS,old.TRUTH,old.PANEL,old.SOURCE_CACHE
read,write=old.read,old.write

def reference_config(arm='B01'):
 if arm not in ('B00','B01'):raise ValueError('Only exact original P43 NoAs/As references')
 return old.config(next(r for r in old.rows() if r['arm']==arm),30)
def old_config(arm=None):return reference_config('B01')
def rows():return [dict(arm=a,row_id=a+'-s'+str(SEED),model_seed=SEED) for a in ARMS]
def config(row,segment=30):
 if row not in rows() or segment!=30:raise ValueError('Only two fixed ownAs20+10 total30 rows')
 inherited=old_config();c=deepcopy(inherited)
 c.update(schema='FcrossRX_quality_risk70_v1',run_id=RUN,**row,launch_owner=OWNER,
  output_root=str(BASE/row['row_id']/'source'),parent_source=inherited['parent_source'],
  checkpoint_sources=[inherited['parent_source']+'/final_ssdg.pth'],
  method='POOL_RANDOM' if row['arm']=='M1' else 'QUALITY_STRAT',
  stage='FcrossRX_fullsource_quality_risk70',
  model_initialization='Strict same legal original P43 As20 student/ownEMA; reset AdamW and new cosine10',
  training_recipe='Original sixL768/CE1.25/sameIQ RDrop/ORD128 SupCon/nativeU batch_neighbor; As64 from fixed shared nominal pool',
  optimizer_policy='RESET AdamW/current CENTER_UNITS/new cosine10; no optimizer inheritance',
  quality_pool=dict(root=str(BASE/'nominal_pool'),N=142080,batch_size=64,updates=2220,
   source='Exact original P43 As30 tail L trace; original epoch/step/named channel streams',
   bins=[3.,8.,15.],measure='signal_to_added_noise_db (not total SNR)',
   nominal_weights='Fixed p_g=N_g/N from this finite nominal training pool, before any update; no V/target',
   one_use=True,extra_proposals=0,per_nonempty_group_min_count=2220,
   insufficient_group='Preserve generated pool; stop before any training, no fallback/retry',
   allocation_seed=20261018070,cache_shared_across_rows=True,
   pool_risk_claim='Fixed-theta unbiased finite-pool group-mean estimator under random within-group allocation; not trajectory-conditional or continuous-population exact unbiasedness',
   M1='Uniform random one-use permutation, ordinary mean CE',
   M2='All groups present each step; floor/ceil N_g/2220 counts, row64, sum p_g mean_g CE',
   satellite_CE_coefficient=.25,satellite_KL_coefficient=.25,
   satellite_KL='Original unweighted same-IQ RDrop KL; not quality-weighted',
   clean_ordinary_raw_U_SupCon_unchanged=True,all_pack_labels_domains_physical_metadata_move=True,
   M3='Not launched; separate bounded risk change only after source evidence'),
  baseline_reference='Reuse P43 original NoAs30 and As30; M1 is required same-cache new control',
  budget_authorization='Two sameAs20+10 totalE30/6660; no new seed; source-qualified good recipe may later totalE50 with matched controls',
  benchmark_context='Exposed benchmark; source-selected mechanism, formal scores never feed promotion',
  goal=dict(fullV_clean_M2_vs_NoAs_min_pp=-.1,pooled_midlow_M2_vs_M1_min_pp=.5,
   source_FAIL_still_all7=True,E50_only_after_source_PASS=True,formal_feedback=False))
 return c

def validate(c):
 row=next((r for r in rows() if r['row_id']==c.get('row_id')),None)
 if row is None or c!=config(row):raise ValueError('Fixed same-parent finite-pool nominal-risk contract differs')
 return c

def make_args(c,device='cuda:0'):
 validate(c);a=old.make_args(old_config(),device)
 a.run_id,a.candidate_id,a.output_dir=RUN,c['row_id'],c['output_root']
 a.baseline_ckpt=c['parent_source']+'/final_ssdg.pth'
 if a.teacher_ckpt or a.pseudo_temporal_mode!='batch_neighbor' or not a.pseudo_temporal_gate:raise ValueError('Original ownEMA/native U rules required')
 return a
