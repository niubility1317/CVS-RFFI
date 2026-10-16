import os,sys,time
from pathlib import Path
import numpy as np
from . import design as d
from .core import periodograms,cell_spectra,analyze

def run():
 c=d.config(d.rows()[0]);d.validate(c)
 if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('CPU-only source G0')
 if d.BASE.exists():raise FileExistsError('Reconcile existing immutable output;no duplicate launch')
 d.BASE.mkdir();out=Path(c['output_root']);out.mkdir();started=time.time()
 d.write(d.BASE/'execution.json',dict(status='RUNNING',pid=os.getpid(),cwd=os.getcwd(),argv=sys.argv,owner=d.OWNER,GPU='',actual_code_commit=(d.ROOT/'release_commit.txt').read_text().strip(),read_at=started))
 try:
  d.write(out/'resolved_config.json',c)
  # Reuse exact validated read-only source loader;never import a model/checkpoint.
  from experiments.cvs_source_repeatcorr.run import source_observations
  x,tx,rx,day,roles=source_observations();psd=cell_spectra(periodograms(x),tx,rx,day);summary,prom=analyze(psd)
  np.savez(out/'source_line_statistics.npz',cell_PSD=psd,cell_prominence_db=prom,shifted_bins=np.asarray(d.BINS))
  summary.update(config=c,roles=roles,wall_seconds=time.time()-started,source_IQ_values_inspected=5040,physical_packet_counts=dict(L=5040,U=0,V=0,held_RX1=0,target=0),source_cells=72)
  d.write(out/'summary.json',summary);d.write(d.BASE/'completion.json',dict(status='ARTIFACTS_COMPLETE',scientific_status=summary['status'],source_only=True,target_access=False,model_forwards=0,checkpoint_loads=0,optimizer_updates=0,new_models=0,wall_seconds=summary['wall_seconds'],automatic_training=False,goal_completed=False))
  print('COMPLETE '+str({k:summary[k] for k in ('status','checks','successful_folds','modal_count','wall_seconds')}),flush=True)
 except Exception as error:
  d.write(d.BASE/'failure.json',dict(status='FAILED',error=repr(error),preserve_outputs=True,no_automatic_retry=True));raise
