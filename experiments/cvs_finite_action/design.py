from pathlib import Path
from experiments.cvs_nocfo_engineering import design as e
ROOT=Path(__file__).resolve().parents[2]
PROJECT,SOURCE,SEEDS,FULL_FP32_POLICY=(getattr(e,k) for k in ('PROJECT','SOURCE','SEEDS','FULL_FP32_POLICY'))
read,write=e.read,e.write
RUN='20261012-phase1-finite-action-transfer-manysig-m3-r01';RELEASE='cvs_finite_action_20261012_r01';OWNER='codex/root/finite-action-transfer-20261012';BASE=Path(PROJECT)/'runs'/RUN
LEVELS=('time_down','t2.pool','id_feat_joint');KINDS=('spectral','timing','noise')
POINTS=dict(spectral=[[-1.5,0.],[1.5,0.],[0.,-1.5],[0.,1.5]],timing=[[-.25,0.],[.25,0.]],noise=[[.03,0.],[.1,0.]])
def rows():return [dict(row_id=f'ACTION-s{s}',model_seed=s) for s in SEEDS]
def config(row):
 if row not in rows():raise ValueError('Unregistered action diagnostic row')
 return dict(run_id=RUN,**row,output=(BASE/row['row_id']).as_posix(),launch_owner=OWNER,source_only=True,target_access=False,identity_updates=0,head_updates=0,checkpoint=(e.f.BASE/f"CE-s{row['model_seed']}"/'source/final_ssdg.pth').as_posix(),teacher_commit=e.TEACHER_EXECUTION_COMMIT,source_data_config=e.f.config(next(r for r in e.f.rows() if r['model_seed']==row['model_seed'])),levels=list(LEVELS),kinds=list(KINDS),points=POINTS,batch_size=256,noise_repeats=4,U_points={'spectral':[1],'timing':[1],'noise':[1]},U_fit=False,U_truth=False,V_forward=0,U_count=56700,L_count=6300,fit_epochs=20,fit_batch_size=256,fit_lr=1e-3,fit_weight_decay=1e-4,hidden_width=128,ridge_relative=1e-3,lowrank=8,coverage_rank=6,source_split_seed=row['model_seed']+271828,fit_seed=row['model_seed']+314159,quality='log complex RMS, power spectral flatness, lag16 normalized coherence; proxies, not calibrated SNR/preamble proof',quality_split='source-only L coherence tertiles define groups; ties assigned by PID rank; boundaries not predictor normalization; all fit means/stds use fit IDs only',splits=['PID80_20',*[f'TX{t}' for t in range(6)],*[f'day{v}' for v in (1,2,3)],*[f'quality{v}' for v in range(3)]],methods=['zero','parameter_linear','bilinear_ridge','lowrank8','MLP'],networks='independent per action and actual level; no shared auxiliary weights',reference='all3 fixed source CE final20, no target-driven choice; eval/no_grad throughout; source encoder has seen all5RX',selection='report all splits/methods/levels/seeds; no target or V choice; no automatic identity training',prelude='N/A protocol-confirmed repeated preamble structure unavailable; lag16 coherence only diagnostic, no prelude loss')
def validate(c):
 r=next((r for r in rows() if r['row_id']==c.get('row_id')),None)
 if r is None or c!=config(r):raise ValueError('Finite action fixed config differs')
 return c
