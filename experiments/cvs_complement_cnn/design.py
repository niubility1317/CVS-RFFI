from copy import deepcopy
from pathlib import Path
from experiments.cvs_quality_film import design as q
from experiments.cvs_stochastic_ensemble import design as s
ROOT=Path(__file__).resolve().parents[2]
PROJECT,SOURCE,FULL_FP32_POLICY,read,write=(getattr(q,k) for k in ('PROJECT','SOURCE','FULL_FP32_POLICY','read','write'))
RUN='20261012-phase1-complement-cnn-source-manysig-s2026092701-r01';RELEASE='cvs_complement_cnn_20261012_r01';OWNER='codex/root/complement-cnn-20261012';BASE=Path(PROJECT)/'runs'/RUN;REFERENCE=s.BASE
SEED=q.SEED;RX=q.RX;ARMS=('WAVE_CNN',)
def rows():return [dict(row_id=f'WAVE_CNN-RX{rx}-s{SEED}',arm='WAVE_CNN',held_rx=rx,model_seed=SEED) for rx in RX]
def config(row):
 if row not in rows():raise ValueError('Unregistered fullwave CNN source row')
 c=deepcopy(q.config(next(r for r in q.rows() if r['arm']=='CE30' and r['held_rx']==row['held_rx'])));c.pop('quality_film')
 c.update(run_id=RUN,**row,output=(BASE/row['row_id']).as_posix(),launch_owner=OWNER,source_U=False,U_cache=None,goal=dict(primary='MIXTURE_CNN_NATIVE3',scope='source_only_not84',feedback_to_training=False),
  cnn=dict(channels=[32,64,128,128],kernels=[9,5,5,3],norm='perpacketGroupNorm8',pool='AvgPool2aftereachconv',flatten_width=2048,hidden=256,feature=160,dropout=.2,head='cosinefixedscale16',input='packetRMSnorm,fixed80:160lag20slopecompensation;no explicit CFO scalar,verified-preamble claim or arbitraryRXinvariance'),
  ensemble=dict(formula='0.5*CNNsoftmaxT1+0.5*fixed3nativeprobabilitymeanT1;allclasses/packets same;no fitting',native_reference=(REFERENCE/f'ENSEMBLE-RX{row["held_rx"]}').as_posix(),models=4),
  scientific_rule=dict(mean_gain_vs_native3=.01,nonnegative_RX=4,worst_RX_noninferiority=.01),source_validation='heldRXV once afterCNNfinal30;native3frozen probabilities exactIDs;no target',
  initialization='fullyrandom fullwaveCNN, no checkpoint inherited',budget_authorization='user scratch30;5newCNN33300updates;matched15native historical99900updates additional,not equalcost advantage')
 return c
def validate(c):
 row=next((r for r in rows() if r['row_id']==c.get('row_id')),None)
 if row is None or c!=config(row):raise ValueError('Fixed fullwave CNN config differs')
 return c
