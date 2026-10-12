from pathlib import Path
from experiments.cvs_e20_response import design as r
ROOT=Path(__file__).resolve().parents[2]
PROJECT,SOURCE,SEEDS,FULL_FP32_POLICY=(getattr(r,k) for k in ('PROJECT','SOURCE','SEEDS','FULL_FP32_POLICY'))
read,write=r.read,r.write
RUN='20261012-phase1-response-identifiability-manysig-m3-r01';RELEASE='cvs_response_identifiability_20261012_r01';OWNER='codex/root/response-identifiability-20261012';BASE=Path(PROJECT)/'runs'/RUN
TAGS=('E20',*r.ARMS)
E40_EXECUTION_COMMIT='acdaf7e47908d8b27fc996d6b1b49787b23f6639'
def rows():return [dict(row_id=f'IDENT-s{s}',model_seed=s) for s in SEEDS]
def response_config(seed):return r.config(next(x for x in r.rows() if x['model_seed']==seed and x['arm']=='RESP_LU'))
def config(row):
 if row not in rows():raise ValueError('Unregistered identifiability row')
 c=response_config(row['model_seed'])
 return dict(run_id=RUN,**row,output=(BASE/row['row_id']).as_posix(),launch_owner=OWNER,source_only=True,target_access=False,identity_updates=0,auxiliary_updates=0,source_contract=SOURCE,model_tags=list(TAGS),E20_commit=r.TEACHER_EXECUTION_COMMIT,E40_commit=E40_EXECUTION_COMMIT,E40_source_run=r.RUN,finite_source_run=c['response']['source_run'],finite_source_commit=r.FINITE_EXECUTION_COMMIT,response_config=c,tasks=c['response']['tasks'],known_parameters=c['response']['known_parameters'],L_count=6300,U_count=56700,U_truth=False,U_fit=False,V_forward=0,batch_size=512,jacobian_samples_per_stratum=128,jacobian_probes=8,jacobian_seed=row['model_seed']+610123,shuffle_seed=row['model_seed']+74129,source_model_packet_occurrences=1512000,model_count=8,record_count=96,strata=['L_fit','L_eval','U_all'],selection='all fixed E20+all7 E40;source-only diagnostic,no training or selection',origin_correction='physical D(a)-D(0); normalized prediction subtract y_mean/y_scale',ablations=['fit_h_mean','fit_q_mean','stratum_shuffle_h','stratum_shuffle_q','original_parameter_linear'],source_split='original PID80_20, fit stats inherited only')
def validate(c):
 row=next((x for x in rows() if x['row_id']==c.get('row_id')),None)
 if row is None or c!=config(row):raise ValueError('Fixed source diagnostic config differs')
 return c
