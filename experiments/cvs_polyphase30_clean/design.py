from pathlib import Path
from experiments.cvs_scratch30_baseline import design as parent
ROOT=Path(__file__).resolve().parents[2]
PROJECT,SOURCE,CAPSULE,TRUTH,CLASSES,SEEDS,FULL_FP32_POLICY,read,write=(getattr(parent,k) for k in ('PROJECT','SOURCE','CAPSULE','TRUTH','CLASSES','SEEDS','FULL_FP32_POLICY','read','write'))
RUN='20261013-phase1-polyphase-clean-manysig-m12-r01'
RELEASE='cvs_polyphase30_clean_20261013_r01'
OWNER='codex/root/polyphase30-clean-20261013'
BASE=Path(PROJECT)/'runs'/RUN
ARMS=('ORDINARY_GLOBAL_GN','ORDINARY_POINT_GN','LOWPASS_POINT_GN','CARRY_POINT_GN')
READOUTS=('NATIVE_HEAD',)
VIEWS=('clean',)
SOURCE_EVIDENCE=Path(PROJECT)/'runs/20261013-source-six-direction-mechanisms-manysig-m3-r03'
LAYOUTS=dict(zip(ARMS,('ordinary_global_GN','ordinary_point_GN','lowpass_point_GN','carry_point_GN')))

def rows():return [dict(row_id=f'{a}-s{s}',arm=a,model_seed=s) for s in SEEDS for a in ARMS]

def config(row):
    if row not in rows():raise ValueError('Unregistered sampling row')
    base=parent.config(next(r for r in parent.rows() if r['arm']=='CE30' and r['model_seed']==row['model_seed']))
    return dict(base,run_id=RUN,**row,output=(BASE/row['row_id']/'source').as_posix(),launch_owner=OWNER,
                source_U=False,U_cache=None,source_validation='source V role checked; no forward, selection or fit',
                test_views=list(VIEWS),readouts=list(READOUTS),
                sampling_layout=LAYOUTS[row['arm']],sampling_layers=['identity.t2.pool','identity.t3'],
                source_evidence=(SOURCE_EVIDENCE/'analysis.json').as_posix(),
                source_evidence_use='all three fixed source sampling applicability; no target metric feedback',
                goal=dict(primary='CARRY_POINT_GN',clean_three_seed_mean=.84,feedback_to_training=False),
                cost_claim='same epochs,updates,L packets; carry retains twice t3 time positions, not samecompute')

def validate(c):
    r=next((r for r in rows() if r['row_id']==c.get('row_id')),None)
    if r is None or c!=config(r):raise ValueError('Fixed sampling config differs')
    return c

def require_source_evidence():
    done=read(SOURCE_EVIDENCE/'completion.json');a=read(SOURCE_EVIDENCE/'analysis.json')
    if done.get('status')!='SOURCE_MECHANISMS_COMPLETE' or done.get('direction_cells')!=18 or done.get('identity_updates')!=0 or done.get('target_access') or not a['directions']['sampling']['eligible_for_preregistered_scratch30']:
        raise ValueError('Full fixed source sampling evidence not available')
    return a['directions']['sampling']
