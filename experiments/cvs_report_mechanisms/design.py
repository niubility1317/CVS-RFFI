from copy import deepcopy
from pathlib import Path
from experiments.cvs_scratch30_baseline import design as b
ROOT=Path(__file__).resolve().parents[2]
PROJECT,SOURCE,CAPSULE,TRUTH,CLASSES,SEEDS,FULL_FP32_POLICY,read,write=(getattr(b,k) for k in ('PROJECT','SOURCE','CAPSULE','TRUTH','CLASSES','SEEDS','FULL_FP32_POLICY','read','write'))
RUN='20261013-phase1-report-mechanisms-clean-manysig-m36-r01';RELEASE='cvs_report_mechanisms_20261013_r01';OWNER='codex/root/report-mechanisms-clean-20261013';BASE=Path(PROJECT)/'runs'/RUN;COUNT=168000;VIEWS=('clean',)
ARMS=('CE_MATCHED','UNIFORM_SSL','META_SSL','SWA_TRAJECTORY','SUBBAND_MODULATION','POLYNOMIAL','AFFINE_IQ','LOG_SPECTRUM','SCALAR_CLOSURE','MATRIX_CLOSURE','TANGENT','C4_ORBIT')
def rows():return [dict(row_id=f'{a}-s{s}',arm=a,model_seed=s) for a in ARMS for s in SEEDS]
def readouts(c):return ('FINAL','SWA') if c['arm']=='SWA_TRAJECTORY' else ('FINAL',)
def config(row):
 if row not in rows():raise ValueError('Unregistered report mechanism row')
 c=deepcopy(b.config(next(r for r in b.rows() if r['model_seed']==row['model_seed'] and r['arm']=='CE30')))
 c.update(run_id=RUN,**row,output=(BASE/row['row_id']/'source').as_posix(),launch_owner=OWNER,source_U=row['arm'] in ('UNIFORM_SSL','META_SSL'),U_cache=b.U_CACHE.as_posix() if row['arm'] in ('UNIFORM_SSL','META_SSL') else None,source_validation='noVforward or performance qualification; fixed30 directlyclean',test_views=['clean'],readouts=['FINAL','SWA'] if row['arm']=='SWA_TRAJECTORY' else ['FINAL'],sampling='CE_MATCHED/SSL LRX/day/TX balanced26/26/26/25/25 rotate feedback; U4otherRX32each; otherarms originalL RX/day completeprivatepermutation',classifier=dict(kind='ordinary_cos',scale=30.,dimension=160,RX_heads=False,routing=False,margin=False,Phase2='discard W and refitfromlegalsupport'),mechanism=dict(arm=row['arm'],branch='boundedadditivejoint160 2sigmoidgain init1; MLP128/GELU/LayerNorm128/160/LayerNorm160' if row['arm'] in ('SUBBAND_MODULATION','POLYNOMIAL','AFFINE_IQ','LOG_SPECTRUM','SCALAR_CLOSURE','MATRIX_CLOSURE') else 'native',tangent_weight=.05,tangent_eta=1.,tangent_actions=['smooth_gain','delay'],C4='nonlinear features mean four exactphase rotations beforeordinarycos; inferenceC4exact, trainingdropout stochastic',SWA_window=list(range(21,31)),SWA_schedule='20cos2e-4to5e-5 thenalternating1e-4/5e-5',source_gate=False),budget_authorization='latestuser scratch30 each/commoncos/noVperformance gate/cleanonly/report mechanisms;all36models6660updates each;SWAextra readout0extraepochs',goal=dict(clean_three_seed_mean=.84,feedback_to_training=False))
 return c
def validate(c):
 r=next((r for r in rows() if r['row_id']==c.get('row_id')),None)
 if r is None or c!=config(r):raise ValueError('Fixedreportmechanism recipe differs')
 return c
