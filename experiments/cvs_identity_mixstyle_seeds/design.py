"""One source-identity-protected feature intervention; single-seed falsification."""
from copy import deepcopy
from pathlib import Path
from experiments.cvs_neighbor_pair50 import design as old
ROOT=Path(__file__).resolve().parents[2]
PROJECT,SOURCE,CAPSULE,TRUTH,CLASSES,FULL_FP32_POLICY,read,write=(getattr(old,k) for k in ('PROJECT','SOURCE','CAPSULE','TRUTH','CLASSES','FULL_FP32_POLICY','read','write'))
SEEDS=old.SEEDS[1:]
RUN='20261014-phase1-identity-mixstyle-oldgate-cos-e50-clean-manysig-s0203-m6-r01'
RELEASE='cvs_identity_mixstyle_seeds_20261014_r01';OWNER='codex/root/identity-mixstyle-seeds-20261014';BASE=Path(PROJECT)/'runs'/RUN
ARMS=('OLD_GATE_COS50','MATCHED_CONTROL','IDENTITY_MIXSTYLE');BASELINE_ARM='OLD_GATE_COS50';PRIMARY='IDENTITY_MIXSTYLE'
PAIRS=[('MATCHED_CONTROL',BASELINE_ARM),(PRIMARY,BASELINE_ARM),(PRIMARY,'MATCHED_CONTROL')]
READOUTS=('ORIGINAL_HEAD',);VIEWS=('clean',);QUERY_COUNT=168000;SOURCE_DOMAINS=15
EPOCHS=50;LABEL_EPOCHS=33;PSEUDO_EPOCHS=17;STEPS=222;UPDATES=11100
BASELINE_EVAL_RUN=old.BASELINE_EVAL_RUN
STYLE_LAYERS=('t1','t2')
def rows():return [dict(row_id=f'{a}-s{s}',arm=a,model_seed=s) for s in SEEDS for a in ARMS]
def training_rows():return [r for r in rows() if r['arm']!=BASELINE_ARM]
def baseline_config(row):return old.baseline_config(dict(arm=BASELINE_ARM,model_seed=row['model_seed']))
def config(row):
 if row not in rows():raise ValueError('Unregistered source MixStyle row')
 prior=baseline_config(row);c=deepcopy(prior)
 c.update(schema='identity_mixstyle_default50_v1',run_id=RUN,**row,method='cvs_identity_mixstyle_seeds',stage='evaluation_reuse' if row['arm']==BASELINE_ARM else 'identity_mixstyle',output_root=(BASE/row['row_id']/'source').as_posix(),launch_owner=OWNER,source_config=prior if row['arm']==BASELINE_ARM else None,model_initialization='historical comparison only' if row['arm']==BASELINE_ARM else 'scratch',readouts=list(READOUTS),classifier_training='default jointly trained only; no separate head fit',classifier='unchanged cosine6x160scale30 normal joint training',
  identity_mixstyle=dict(enabled=row['arm']==PRIMARY,layers=list(STYLE_LAYERS),p=.5,alpha=.1,strength=1.,eps=1e-6,pool='L_s only in normal main supervised forward',pairing='same TX/day; different RX and actual physical ID',fallback='identity',seed_policy='private NumPy: model_seed+95001+main_call*701+layer*1709',U_intervention=False,teacher_intervention=False,inference_intervention=False,extra_forward_calls=0,extra_parameters=0),
  source_falsification=dict(final_L_CE_max=.5,minimum_layer_changed_ratio=.25,minimum_actual_source_z_L2=1e-6,decision='both new controls final LCE<=.5 and both layers changed>=25% over fixed mainL packet calls and actual source z differs; later seed expansion depends only on these source facts, not target',additional_seed_launch=False),
  goal=dict(primary=PRIMARY,readout='ORIGINAL_HEAD',clean_three_seed_mean=.84,feedback_to_training=False),hypothesis_basis='Frozen source t1/t2 RX-conditioned mean/std variations and identity-safe intervention into actual joint160; run 20261014-phase1-source-style-profile-manysig-s01-m1-r01',no_PAIRS_Sinc_TIME_stacking=True)
 c.update(source_qualification_run='20261014-phase1-identity-mixstyle-oldgate-cos-e50-clean-manysig-s01-m3-r01',source_expansion_decision_commit='9e79dc413967aee78fa87e5d0f6c7408c44e4efc',first_seed_retraining=False,combined_three_seed_report='merge corresponding completed firstseed results without model selection; same p/alpha/strength/layers/recipe')
 c.pop('feature_extractor',None)
 return c
def validate(c):
 row=next((r for r in rows() if r['row_id']==c.get('row_id')),None)
 if row is None or c!=config(row) or c['epochs']!=50 or c['readouts']!=['ORIGINAL_HEAD']:raise ValueError('Fixed source MixStyle matrix differs')
 return c
def make_args(c,device='cuda:0'):
 validate(c)
 if c['arm']==BASELINE_ARM:raise ValueError('Historical baseline cannot be retrained')
 a=old.old.make_args(baseline_config(c),device);a.output_dir=c['output_root'];a.run_id=RUN;a.candidate_id=c['row_id'];a.seed=c['model_seed']
 if a.use_mixstyle:raise ValueError('Original native MixStyle must remain off')
 return a
