"""Source availability only; subsequent training requires a separate E20 plan."""
from pathlib import Path
from experiments.cvs_sigtime_heldcredit import design as previous

ROOT=Path(__file__).resolve().parents[2]
PROJECT,SOURCE_RELEASE,ORDINALS,HELD_ORDINALS,SOURCE_RX=(getattr(previous,k) for k in
 ('PROJECT','SOURCE_RELEASE','ORDINALS','HELD_ORDINALS','SOURCE_RX'))
write=previous.write
RUN='20261015-phase1-SIGTIME-crossRX-feature-sharpness-sourceL-manysig-s2026092701-r01'
RELEASE='cvs_SIGTIME_crossRX_feature_sharpness_sourceL_20261015_r01'
OUTPUT=PROJECT+'/runs/'+RUN+'/source_diagnostic'
MODES=('ZERO','CLEAN_CE','GLOBAL_GAP','CROSS_GAP')
GATE=dict(minimum_active_cross_cohorts=2,cross_positive_gap_min=1e-4,
          auxiliary_gradient_min=1e-8,minimum_distinct_cross_global_cohorts=2,
          cross_global_gradient_relative_difference_min=.05,
          minimum_auxiliary_Sinc_kernel_change_cohorts=2,
          interpretation='Source availability and distinct signal only; no efficacy/novelty claim or target selection')

def request():
 return dict(run_id=RUN,output_root=OUTPUT,source_release=SOURCE_RELEASE,
  source_role='L_s',model_seed=2026092701,source_row='BASE-s2026092701',
  coordinate='CENTER_25K',scale_hz=25000.,modes=list(MODES),
  anchor_ordinals=list(ORDINALS),held_ordinals=list(HELD_ORDINALS),
  rho=.05,coefficient=.1,normalize_eps=1e-12,source_falsification=GATE,
  analysis_code_commit='release_commit.txt')

def validate(c):
 expected=request();expected['analysis_code_commit']=c.get('analysis_code_commit')
 if c!=expected or not isinstance(c['analysis_code_commit'],str) or len(c['analysis_code_commit'])!=40:
  raise ValueError('Fixed cross-RX sharpness request or actual Git commit differs')
 return c
