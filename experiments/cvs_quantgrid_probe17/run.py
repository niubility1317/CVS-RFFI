"""CPU2 complete legal fit-L measurement;0model/checkpoint/NN/IQ transformations."""
import gzip,json,os,sys,time
from pathlib import Path
import numpy as np
import torch
from . import design as d
from .quantgrid import estimate
from .statistics import summarize

def run():
 c=d.validate(d.config())
 if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('CPU-only statistics')
 if d.BASE.exists():raise FileExistsError('Reconcile own immutable output,no duplicate launch')
 d.BASE.mkdir();started=time.perf_counter();torch.set_num_threads(2)
 d.write(d.BASE/'execution.json',dict(status='RUNNING',pid=os.getpid(),cwd=os.getcwd(),argv=sys.argv,owner=d.OWNER,CUDA_VISIBLE_DEVICES='',code_commit=(d.ROOT/'release_commit.txt').read_text().strip(),read_at=time.time(),CPU_threads=2,network_forwards=0));d.write(d.BASE/'resolved_config.json',c)
 try:
  # The first step is synthetic-only;no model/checkpoint exists in this diagnostic.
  rng=np.random.default_rng(d.SEED);x=np.rint(rng.normal(size=(2,256))/.03125)*.03125;smoke=estimate(x)
  if not smoke['observable'] or abs(smoke['q']-.03125)>.00001:raise ValueError('Synthetic noquery lattice smoke failed')
  d.write(d.BASE/'synthetic_noquery_smoke.json',dict(status='PASS',real_IQ=0,checkpoint_reads=0,NN=0));print('SYNTHETIC_SMOKE PASS 0model/CP/IQ/NN',flush=True)
  carrier=d.read(d.SOURCE);contract=d.verify_contract(d.read(d.REFERENCE),carrier)
  from .source import load
  L=load(contract,carrier);values=np.asarray(L['x'].double().tolist(),dtype=np.float64);d.write(d.BASE/'source_contract.json',contract);d.write(d.BASE/'data_roles.json',dict(only_fitL5040_consumed=True,U_IQ_or_truth=0,V_values=0,held_RX1_values=0,target=0,carrier_L6300_and_unusedV_oldfeatures_deserialized=True,old_cached_features_consumed=False,normalization='Original cache representation unchanged',checkpoint_reads=0))
  rows=[]
  with gzip.open(d.BASE/'packets.jsonl.gz','xt',encoding='utf8',compresslevel=3) as f:
   for i,x in enumerate(values):
    result=estimate(x);r=dict(physical_id=L['ids'][i],TX=int(L['y'][i]),RX=int(L['rx'][i]),day=int(L['day'][i]),measurement=result);rows.append(r);f.write(json.dumps(r,allow_nan=False)+'\n')
    if (i+1)%64==0 or i==5039:
     f.flush();d.write(d.BASE/'progress.json',dict(processed=i+1,total=5040,observable=sum(v['measurement']['observable'] for v in rows),NN=0,updates=0,source_only=True,target_access=False));print('SOURCE '+str(i+1)+'/5040',flush=True)
  s=summarize(rows);s.update(status='ARTIFACTS_COMPLETE',source_only=True,target_access=False,actual_source_IQ_packets=5040,source_cells=72,checkpoint_reads=0,model_epochs=0,optimizer_updates=0,network_forwards=0,head_fit_updates=0,automatic_training=False,Ag_efficacy_proved=False,goal_complete=False,wall_seconds=time.perf_counter()-started);d.write(d.BASE/'summary.json',s);d.write(d.BASE/'completion.json',dict(status='ARTIFACTS_COMPLETE',scientific_status=s['scientific_status'],wall_seconds=s['wall_seconds'],source_only=True,target_access=False,NN=0,updates=0));print('COMPLETE '+s['scientific_status'],flush=True)
 except BaseException as e:
  d.write(d.BASE/'failure.json',dict(status='FAILED',error=repr(e),preserve_outputs=True,no_automatic_retry=True));raise
if __name__=='__main__':
 if sys.argv[1:]:raise ValueError('Fixed source study,no overrides')
 run()
