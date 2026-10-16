from pathlib import Path
from experiments.cvs_f_practical_source20 import design as prior
ROOT=Path(__file__).resolve().parents[2]
PROJECT,SOURCE,SEED,DRAW_SEED,SCENES,read,write=(getattr(prior,k) for k in ('PROJECT','SOURCE','SEED','DRAW_SEED','SCENES','read','write'))
RUN='20261016-diagnostic-F-phase-origin-source-RX1-s2026092701-m2-r02'
RELEASE='cvs_F_phase_origin_source_RX1_20261016_r02'
OWNER='codex/root/F-phase-origin-source-RX1-20261016'
BASE=Path(PROJECT)/'runs'/RUN
PREVIOUS=Path(PROJECT)/'runs'/prior.RUN
CONDITIONS=('phase','delay','phase_delay')

def rows():return prior.rows()
def config(row):
    c=prior.config(row)
    return dict(c,schema='F_phase_origin_source_factor_v1',run_id=RUN,launch_owner=OWNER,output_root=(BASE/row['row_id']).as_posix(),
        conditions=list(CONDITIONS),raw_condition='unitRMS原IQ once',prior_source_predictions=PREVIOUS.as_posix(),
        delay_samples=24,boundary='reflect',phase='unit complex phase of largest-magnitude tap in original full propagation kernel; same as previous flat_H',
        draw_seed=DRAW_SEED,stream='Reuse exact previous source namespace/scene/opaqueID; no new model/training/target random draw',
        treatment='2x2 raw/phase/delay/phase_delay; no new noise/geometry/receiver compensation, full-state metadata only defines cohorts',
        channel_route='Synthetic source factors only; original residual used for H phase and fixed cohorts, no practical/test replacement',
        expected_packet_forwards=23942,expected_derivative_calls=0,
        selection='A0 fixed primary, failed W30 negative control; no model/strength/scene selection or automatic training',
        scientific_scope='Source attribution only; if all A0 accuracy and raw-correct retention effects abs<1pp abandon major-factor interpretation')
def validate(c):
    row=next((r for r in rows() if r['row_id']==c.get('row_id')),None)
    if row is None or c!=config(row):raise ValueError('Fixed source phase/origin design differs')
def prerequisite():
    x=read(PREVIOUS/'completion.json')
    if (x['status']!='SOURCE_F_PRACTICAL_DIAGNOSTIC_COMPLETE' or x['packet_forwards']!=168844
        or x['optimizer_updates'] or x['target_access'] or not x['source_only']):raise ValueError('Completed original source attribution required')
    return x
