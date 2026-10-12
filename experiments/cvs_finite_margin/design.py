from pathlib import Path
from experiments.cvs_wave_geometry import design as w
ROOT=Path(__file__).resolve().parents[2]
PROJECT,SOURCE,FULL_FP32_POLICY,read,write,SEED,RX=(getattr(w,k) for k in ('PROJECT','SOURCE','FULL_FP32_POLICY','read','write','SEED','RX'))
RUN='20261012-phase1-finite-margin-source-manysig-s2026092701-r01'
RELEASE='cvs_finite_margin_20261012_r01';OWNER='codex/root/finite-margin-20261012';BASE=Path(PROJECT)/'runs'/RUN
ARMS=('CLEAN_REPLAY','ACTION_CE','FINITE_MARGIN')
ACTIONS=[['spectral',[1.5,0.]],['spectral',[-1.5,0.]],['timing',[.25,0.]],['timing',[-.25,0.]]]
def rows():return [dict(row_id=f'{a}-RX{r}-s{SEED}',arm=a,held_rx=r,model_seed=SEED) for r in RX for a in ARMS]
def config(row):
 if row not in rows():raise ValueError('Unregistered finite margin row')
 return dict(run_id=RUN,**row,output=(BASE/row['row_id']).as_posix(),launch_owner=OWNER,source_contract=SOURCE,source_only=True,target_access=False,checkpoint_sources=[],source_loader_config=w.config(w.rows()[0])['source_loader_config'],epochs=20,steps_per_epoch=222,updates=4440,batch_size=128,extra_batch_size=32,peak_lr=.0002,min_lr=.000001,weight_decay=.0001,gradient_clip=10.,loss_weight=.5,actions=ACTIONS,source_U=False,EMA=False,satellite=False,auxiliary=False,explicit_CFO_identity_feature=False,initialization='whole native CVS identity from scratch; no weights or teacher',checkpoint_selection='final_only',extra_mode='eval with input/parameter gradient; no dropout or BN updates',main_mode='train clean CE',margin='logsumexp_c(-min_a(f_y(A_a x)-f_c(A_a x))); clean and four actual views; uniform all competitors; finite set only',compute_control='all arms generate four actions and execute one extra160-packet eval forward; CLEAN_REPLAY forwards five identical clean copies',source_validation='heldRX V only after epoch20; no updates; source clean+four finite operations',held_rx_excluded_from_all_fitting=True,scientific_rule=dict(mean_gain_ACTION_CE=.005,nonnegative_RX=4,clean_mean_noninferiority_REPLAY=.005,worst_RX_noninferiority_ACTION_CE=.01,finite_stress_mean_noninferiority_ACTION_CE=0.),sampling='same RX/day-balanced stream on four RX only; main128 plus its first32 for extra views',budget='fixedE20 4440 updates; five RX x three arms, 15 fresh native identity models')
def validate(c):
 row=next((r for r in rows() if r['row_id']==c.get('row_id')),None)
 if row is None or c!=config(row):raise ValueError('Finite margin fixed config differs')
 return c
