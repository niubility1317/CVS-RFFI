"""Finalize already-generated R02 bank metadata without reading IQ payload or resimulation."""
from collections import Counter
import gzip,json,time
from pathlib import Path
import numpy as np
from . import design as d
from .bank import FIELDS,bins
TIMERS=('raw_dataset_cache_cpu_wall_seconds','simulation_adapter_cpu_wall_seconds','IQ_cache_write_cpu_wall_seconds','metadata_write_cpu_wall_seconds','wall_seconds')

def source_L_role_ids(contract):
 roles=contract['role_ids']
 if set(roles)!={'L_s','U_s','V'} or {k:len(v) for k,v in roles.items()}!={'L_s':6300,'U_s':56700,'V':27000}:raise ValueError('Existing actual source minimal role handles differ')
 return roles['L_s']

def read_iq_header(stream,file_bytes):
 version=np.lib.format.read_magic(stream)
 if version!=(1,0):raise ValueError('Expected original open_memmap NPY v1 header')
 shape,fortran,dtype=np.lib.format.read_array_header_1_0(stream);offset=stream.tell();payload=int(np.prod(shape))*dtype.itemsize
 if shape!=(252000,2,256) or fortran or dtype!=np.dtype('float32') or file_bytes!=offset+payload:raise ValueError('Completed IQ header/size differs')
 return dict(shape=list(shape),dtype=str(dtype),fortran_order=bool(fortran),header_bytes=offset,payload_bytes=payload,file_bytes=int(file_bytes),payload_read=False)

def validate_metadata(a):
 expected={'TX','RX','day','domain','bin','scene','state','lock','signal_to_added_noise_db','quality_snr_db','cumulative_epoch','replica','context_step','source_L_index','sample_ids'}
 if set(a)!=expected or any(v.shape!=((252000,5) if k=='sample_ids' else (252000,)) for k,v in a.items()):raise ValueError('Completed252000 metadata schema differs')
 if not np.array_equal(a['source_L_index'],np.tile(np.arange(6300),40)) or not np.array_equal(a['bin'],bins(a['signal_to_added_noise_db'])) or not np.isfinite(a['quality_snr_db']).all():raise ValueError('Original complete bank indexing/bin metadata differs')
 counts=np.bincount(a['bin'],minlength=4)
 if (counts<=0).any():raise ValueError('Completed bank has empty fixed bin')
 for key in ('TX','RX','day','domain','sample_ids'):
  v=a[key].reshape((40,6300,5) if key=='sample_ids' else (40,6300))
  if not np.all(v==v[0]):raise ValueError('Same sourceL identity metadata must repeat across40contexts')
 if set(a['TX'].tolist())!=set(range(6)) or set(a['RX'].tolist())!=set(d.RX) or set(a['day'].tolist())!={1,2,3} or set(a['domain'].tolist())!=set(range(15)):raise ValueError('Original source TX/RX/day/domain support differs')
 for ci,c in enumerate(d.bank_contexts()):
  sl=slice(ci*6300,(ci+1)*6300)
  if not np.all(a['cumulative_epoch'][sl]==c['cumulative_epoch']) or not np.all(a['replica'][sl]==c['replica']) or not np.all(a['context_step'][sl]==c['cumulative_step']):raise ValueError('Original40 legal contexts differ')
 return counts

def count_physics(stream):
 n=0;sessions={}
 for line in stream:
  row=json.loads(line)
  if row['bank_index']!=n:raise ValueError('Closed physics index sequence differs')
  p=row['physics'];session=p['session_id']
  if session not in sessions:sessions[session]=p['receiver']
  n+=1
 if n!=252000:raise ValueError('Closed physics metadata count differs')
 return n,sessions

def finalize_existing_bank():
 started=time.perf_counter()
 if set(p.name for p in d.BANK.iterdir())!={'iq.npy','metadata.npz','physics.jsonl.gz'}:raise ValueError('Only original three completed bank files without manifest may finalize')
 files={name:dict(bytes=(d.BANK/name).stat().st_size,mtime=(d.BANK/name).stat().st_mtime) for name in ('iq.npy','metadata.npz','physics.jsonl.gz')}
 with (d.BANK/'iq.npy').open('rb') as stream:header=read_iq_header(stream,files['iq.npy']['bytes'])
 with np.load(d.BANK/'metadata.npz',allow_pickle=False) as f:a={k:f[k].copy() for k in f.files}
 counts=validate_metadata(a)
 with gzip.open(d.BANK/'physics.jsonl.gz','rt',encoding='utf8') as stream:physics_count,sessions=count_physics(stream)
 roles=d.read(d.SOURCE);role_ids=source_L_role_ids(roles);strata=Counter(zip(a['TX'].tolist(),a['RX'].tolist(),a['day'].tolist(),a['bin'].tolist(),a['scene'].tolist(),a['state'].tolist(),a['lock'].tolist()));channel_strata=Counter(zip(a['RX'].tolist(),a['day'].tolist(),a['scene'].tolist(),a['state'].tolist(),a['lock'].tolist()))
 generation=dict(calls=None,packets=252000,session_count=len(sessions),sessions=sessions,strata=[dict(RX=k[0],day=k[1],scene=k[2],state=k[3],lock=k[4],count=n) for k,n in sorted(channel_strata.items())],all_original_source_RX=list(d.RX),dropped=0,confidence_filter=False,ordinary_before_satellite=False,formal_test_changed=False,counter_status='packets/sessions/strata reconstructed from closed R02 metadata;original API-call counter unavailable')
 value=dict(status='SOURCE_L_BANK_COMPLETE',run_id=d.RUN,role='L',source_role_key='L_s',source_role_counts={k:len(v) for k,v in roles['role_ids'].items()},count=252000,source_L=6300,each_L_views=40,replicas_per_epoch=4,contexts=d.bank_contexts(),shape=[252000,2,256],dtype='float32',IQ_bytes=252000*2*256*4,bin_field='signal_to_added_noise_db',bin_edges=list(d.BIN_EDGES),bin_counts=counts.tolist(),nominal_bin_mass=(counts/252000).tolist(),sample_id_fields=list(FIELDS),physics_file='physics.jsonl.gz',index_file='metadata.npz',IQ_file='iq.npy',source_contract=str(d.SOURCE),role_ids=role_ids,channel=dict(metadata_recovered=True,original_reference_channel=d.reference_config(d.rows()[0],30)['channel'],generation_release='cvs_FcrossRX_qualityRisk_E30_20261018_r02',no_physical_reprocessing=True),channel_generation=generation,source_L_index_counts=np.bincount(a['source_L_index'],minlength=6300).tolist(),strata=[dict(TX=k[0],RX=k[1],day=k[2],bin=k[3],scene=k[4],state=k[5],lock=k[6],count=n) for k,n in sorted(strata.items())],**{k:None for k in TIMERS},timing='Original bank CPU/wall timers lost before R02 manifest assignment;null,not zero or reconstructed duration',timing_status='UNAVAILABLE_R02_FINALIZER_EXCEPTION',recovered_existing_bank=True,recovery_release=d.RELEASE,generation_release='cvs_FcrossRX_qualityRisk_E30_20261018_r02',finalization_cpu_wall_seconds=time.perf_counter()-started,original_file_stats=files,IQ_header=header,physics_metadata_rows=physics_count,IQ_payload_reads=0,bank_files_rewritten=False,no_additional_propagation=True,propagation_simulations=252000,additional_propagation_simulations=0,NN=0,checkpoint_reads=0,U_truth_used=False,U_iterations=0,target_access=False,quality_rejection=False,retries=0,bank_online_risk_equivalence_claim=False)
 d.write(d.BANK/'complete.json',value);return value
