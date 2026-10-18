import os,sys,time,subprocess,traceback
from pathlib import Path
import numpy as np
from . import design as d
from .core import load_observations,evaluate

def main():
 subprocess.run([sys.executable,'-m','experiments.cvs_iqband_source51.checks'],check=True,stdout=subprocess.DEVNULL)
 if d.BASE.exists():raise FileExistsError('Immutable output root exists;reconcile')
 d.BASE.mkdir();started=time.perf_counter();d.write(d.BASE/'checks.json',dict(status='PASS',tests=7,real_checkpoint='NOT_APPLICABLE_NO_MODEL_CONSUMED',IQ_reads=0,NN=0,target_access=False));d.write(d.BASE/'active.json',dict(pid=os.getpid(),start_ticks=int(Path('/proc',str(os.getpid()),'stat').read_text().split()[21]),cwd=os.getcwd(),argv=sys.argv,CUDA_VISIBLE_DEVICES=os.environ.get('CUDA_VISIBLE_DEVICES'),read_at=time.time()))
 try:
  d.write(d.BASE/'resolved_config.json',dict(config=d.config(),code_commit=(d.ROOT/'release_commit.txt').read_text().strip(),source_roles='EXACT_MATCH_REUSED_P50_DERIVED_PERIODIGRAMS',python=sys.executable,CWD=os.getcwd(),GPU=None,NN=0,IQ_reads=0,leo_practical_file='NOT_IMPORTED_NO_CHANNEL'))
  o=load_observations(d.OBS,d.INPUT_CONFIG);result,fitted=evaluate(o['L'],o['V'],o['L0'],o['L1']);wall=time.perf_counter()-started;result.update(wall_seconds=wall,source_derived_packets=33300,source_derived_real_values=16916400,source_fold_fits=72,IQ_reads=0,checkpoint_reads=0,NN=0,training_updates=0,teacher_forwards=0,head_fits=0,new_IQ_views=0,source_RX_holdout=None,target_access=False)
  np.savez_compressed(d.BASE/'fitted_statistics.npz',**fitted);d.write(d.BASE/'result.json',result);print(result['decision'],flush=True);subprocess.run([sys.executable,str(Path(__file__).with_name('recount.py')),str(d.BASE)],check=True)
  d.write(d.BASE/'completion.json',dict(status='SOURCE_ANALYZED',decision=result['decision'],source_qualified=result['source_qualified'],source_derived_packets=33300,NN=0,IQ_reads=0,checkpoint_reads=0,training_updates=0,target_access=False,source_wall_seconds=wall,wall_seconds=time.perf_counter()-started,independent_recount='PASS'))
 except Exception as exc:
  d.write(d.BASE/'failure.json',dict(error_type=type(exc).__name__,error=str(exc),traceback=traceback.format_exc(),preserve_outputs=True));raise
if __name__=='__main__':main()
