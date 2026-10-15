"""Metadata-only overlay; all training/model code stays in the original release."""
from pathlib import Path
from experiments.cvs_sigtime_rdrop_legacy import design as original
ROOT=Path(__file__).resolve().parents[2]
PROJECT,SOURCE,read,write=(getattr(original,k) for k in ('PROJECT','SOURCE','read','write'))
RUN='20261015-phase1-originalSIGTIME-rdrop-e50-technical-clean-manysig-m4-r02'
RELEASE='cvs_originalSIGTIME_rdrop_e50_technical_20261015_r02'
OWNER='codex/root/original-SIGTIME-RDrop-technical-20261015'
ORIGINAL_RUN='20261015-phase1-originalSIGTIME-rdrop-e20e50-clean-manysig-m18-r01'
ORIGINAL_RELEASE=PROJECT+'/releases/cvs_originalSIGTIME_rdrop20_50_20261015_r01'
MODEL_COMMIT='b3ac442e2d33a5b315fb3251c30e0ad7083566bc'
ORIGINAL_BASE=Path(PROJECT)/'runs'/ORIGINAL_RUN
BASE=Path(PROJECT)/'runs'/RUN
REMAINING=('E50_RDROP-s2026092702','E50_ORIGINAL-s2026092703','E50_TWO_CE-s2026092703','E50_RDROP-s2026092703')
OLD_SCORE_ROWS=('E50_RDROP-s2026092701','E50_ORIGINAL-s2026092702','E50_TWO_CE-s2026092702')
ALL_ORIGINAL_ROWS=[dict(r) for r in original.rows()]
ORIGINAL_OWNER='codex/root/original-SIGTIME-RDrop-20261015'
def rows():return [dict(r) for rid in REMAINING for r in ALL_ORIGINAL_ROWS if r['row_id']==rid]
def activate():
 # Only this process changes metadata. Original release files and running
 # original workers are untouched; ROOT remains the original model-code root.
 original.RUN=RUN;original.BASE=BASE;original.OWNER=OWNER;original.rows=rows
 return original
def config(row):return activate().config(row)
def request():return dict(run_id=RUN,original_run_id=ORIGINAL_RUN,original_release=ORIGINAL_RELEASE,model_code_commit=MODEL_COMMIT,dependency_release=original.DEPENDENCY_RELEASE,fold_release=original.FOLD_RELEASE,rows=[config(r) for r in rows()],old_score_rows=list(OLD_SCORE_ROWS),source_gate_reference=(ORIGINAL_BASE/'source_gates.json').as_posix(),scientific_primary='E50_RDROP',no_model_restarts=True)
def validate(value):
 if value!=request():raise ValueError('Exact four unstarted E50 rows and unchanged recipe required')
 return value
def totals():return dict(models=4,epochs=200,optimizer_steps=44400,extra_L_identity_forwards=4262400,U_observations=3855600,clean_decisions=672000,metric_records=72)
