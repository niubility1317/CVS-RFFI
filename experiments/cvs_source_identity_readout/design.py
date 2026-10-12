from pathlib import Path
from experiments.cvs_source_conditional_geometry import design as old
ROOT=Path(__file__).resolve().parents[2]
PROJECT,SOURCE,SEEDS,FULL_FP32_POLICY=(getattr(old,k) for k in ('PROJECT','SOURCE','SEEDS','FULL_FP32_POLICY'))
read,write=old.read,old.write
RUN='20261012-phase1-e20-identity-readout-manysig-m3-r01'
RELEASE='cvs_source_identity_readout_20261012_r01';OWNER='codex/root/source-identity-readout-20261012'
BASE=Path(PROJECT)/'runs'/RUN

def rows():return [dict(row_id=f'READOUT-s{s}',model_seed=s) for s in SEEDS]

def config(row):
    if row not in rows():raise ValueError('Unregistered identity readout row')
    prior=old.config(next(r for r in old.rows() if r['model_seed']==row['model_seed']))
    return dict(run_id=RUN,**row,output=(BASE/row['row_id']).as_posix(),launch_owner=OWNER,parent_build_config=prior['parent_build_config'],checkpoint=prior['checkpoint'],source_only=True,target_access=False,identity_updates=0,head_updates=0,source_role='L_s',L_count=6300,U_forward=0,V_forward=0,
        levels=['time_down','t2.pool','id_feat_joint'],map_summary='concatenate time mean and population std; final native joint160 unchanged',input_policies=['historical_explicit_CFO','no_explicit_CFO'],CFO_identity_coordinate=27,CFO_policy='zero explicit CFO slot, keep width29 and finite per-packet compensation; raw IQ retains residual information',
        batch_size=128,model_forwards=100,L_model_occurrences=12600,classes=list(range(6)),receivers=[1,3,4,6,8],days=[1,2,3],cell_count=90,packets_per_cell=70,
        crossfit_folds=3,crossfit_seed=row['model_seed']+161803,
        crossfit_scope='within each TX/RX/day cell sort PID, fixed private permutation then indexmod3; query held RX and fold, fit excludes that RX and all query-fold PIDs',
        readouts=['raw_merged_prototype','raw_RX_separate','stable5_merged_prototype','stable5_RX_separate'],prototype_metric='cosine; common fitted centering; RX-separate logmeanexp over equal4RX prototypes then all6TX compete',temperature=.1,
        stable_projection=dict(alpha=1.,beta=1.,shrinkage=.1,eps_relative=1e-6,max_rank=5,PSD='biased finite-population outer products; Cholesky generalized eigenproblem'),
        covariance_scope='balanced TX/RX means with equal day cell weights; total=global TX+global RX+TXxRX interaction+within TX/RX including day variation; rank(TX)<=5',checkpoint_selection='all three user-fixed original pureCE E20; no target ranking',
        claim_scope='frozen encoder saw all five source RX and allL; L-crossfit readout transfer only, not unseen-RX generalization or heldout-encoder evaluation',
        output_policy='aggregate scatters/eigenvalues, per-cell readout/confusions and metrics; no IQ or sample features exported',V_scope=prior['V_scope'])

def validate(c):
    row=next((r for r in rows() if r['row_id']==c.get('row_id')),None)
    if row is None or c!=config(row):raise ValueError('Frozen readout config differs')
    return c
