from pathlib import Path
from experiments.cvs_stage2_teacher_factorial import design as parent
ROOT=Path(__file__).resolve().parents[2]
PROJECT=parent.PROJECT;SOURCE=parent.SOURCE;SEEDS=parent.SEEDS
RUN='20261011-phase1-rx-iq-intervention-manysig-m3-r01'
RELEASE='cvs_rx_intervention_audit_20261011_r01'
BASE=Path(PROJECT)/'runs'/RUN
OWNER='codex/root/rx-iq-intervention-20261011'
read,write=parent.read,parent.write

def rows():return [dict(row_id=f'E200-s{s}',model_seed=s) for s in SEEDS]

def config(row):
 if row not in rows():raise ValueError('Unregistered IQ audit')
 pc=parent.config(next(r for r in parent.rows() if r['arm']=='O_CE' and r['model_seed']==row['model_seed']))
 return dict(run_id=RUN,**row,output=(BASE/row['row_id']).as_posix(),parent_build_config=pc,source_checkpoint=pc['parent_checkpoint'],source_only=True,target_access=False,identity_updates=0,launch_owner=OWNER,
  intervention=dict(observation='rectangular/DCT1to8',crossfit='other five TX AND other two days',scales=[.5,1.],RX_shifts=[1,2,3,4],variants=[dict(name='physical_K33',taps=33,kind='physical'),dict(name='physical_K17',taps=17,kind='physical'),dict(name='rotated_K33',taps=33,kind='rotated')],coefficient_control='roll DCT coordinates by4; coefficient norm matched before clipping, not a matched FIR-energy control',max_amplitude_dB=3.,filter='Hann-truncated centered complex FIR; zero-padded linear convolution; same256 crop; per-packet RMS preserved',window='nonperiodic Hann on FIR taps',primary='physical_K33',gradient_sample='first packet of every L TX/day/RX cell; 90 balanced packets',gradient_scope='dense encoder/fusion-head only; excludes frequency coordinates with different units; norms and cosine, not optimizer steps'),
  counts=dict(L=6300,U_used=0,V_used=0),selection='report all24 fixed views and clean for every seed; no V or target selection; no automatic training launch',
  interpretation='L classifier preservation is a diagnostic proxy, not proof of physical identity invariance; parent has seen all5 sourceRX')

def validate(c):
 r=next((r for r in rows() if r['row_id']==c.get('row_id')),None)
 if r is None or c!=config(r):raise ValueError('IQ diagnostic config differs')
