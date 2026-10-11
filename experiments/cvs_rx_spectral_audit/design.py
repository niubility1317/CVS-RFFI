from pathlib import Path
from experiments.cvs_stage2_teacher_factorial import design as parent

ROOT=Path(__file__).resolve().parents[2]
PROJECT=parent.PROJECT; SOURCE=parent.SOURCE
RUN='20261011-phase1-rx-spectral-crossfit-manysig-m1-r01'
RELEASE='cvs_rx_spectral_audit_20261011_r01'
BASE=Path(PROJECT)/'runs'/RUN
OWNER='codex/root/rx-spectral-audit-20261011'
read,write=parent.read,parent.write

def rows():return [dict(row_id='L6300-spectral-crossfit',model_seed=None)]

def config(row):
 if row not in rows():raise ValueError('Unregistered spectral diagnostic')
 pc=parent.config(next(r for r in parent.rows() if r['arm']=='O_CE' and r['model_seed']==parent.SEEDS[0]))
 return dict(run_id=RUN,**row,output=(BASE/row['row_id']).as_posix(),preparation=pc['preparation'],source_contract=SOURCE,
  launch_owner=OWNER,identity_updates=0,checkpoint_sources=[],target_access=False,source_only=True,
  fit_role='L_s only',role_count=6300,TX_count=6,RX_count=5,day_count=3,cell_count=70,
  observation=dict(samples=256,FFT_bins=256,pooled_bins=32,log_floor_relative=1e-6,windows=['rectangular','hann'],coordinates=['centered_log_PSD32','DCT1to8'],primary='rectangular/DCT1to8'),
  crossfit='18 folds: estimate on other five TX AND other two days; test mean RX contrast of excluded TX/day',
  controls=['zero_contrast','reverse_contrast','four_cyclic_RX_permutations','split_cell_replicate_noise'],
  selection='report all fixed observations; no V/target selection or automatic identity training',
  interpretation='RX-associated response may combine receiver, propagation and acquisition effects; not a calibrated receiver transfer function',
  seed_notes='deterministic IQ-only diagnosis of fixed shared L split; no identity model, training seed or random fit')

def validate(c):
 if c!=config(rows()[0]):raise ValueError('Diagnostic configuration differs')
