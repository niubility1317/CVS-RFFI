"""Known immutable post-As backend and two registered frozen checkpoints only."""
import importlib.util,sys,types
from pathlib import Path
HERE=Path(__file__).resolve().parents[2]
def bootstrap():
 backend=Path('/home/szu2070436088/2510044040/CV-SincNet/releases/cvs_FcrossRX_noisepair_sourceRX1_E20_20261017_r01')
 if (backend/'release_commit.txt').read_text().strip()!='ee305facac7874bfc1340291ee5360a3665621d9':raise ValueError('Wrong immutable post-As model backend')
 spec=importlib.util.spec_from_file_location('postAs_backend',backend/'experiments/cvs_f_noise_pair20/runner.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);m.bootstrap()
 package=types.ModuleType('experiments.cvs_f_postas_noise_probe');package.__path__=[str(HERE/'experiments/cvs_f_postas_noise_probe')];sys.modules['experiments.cvs_f_postas_noise_probe']=package
 from experiments.cvs_f_postas_noise_probe import design as d
 return d
if __name__=='__main__':
 args=sys.argv[1:]
 if args and not (len(args)==2 and args[0]=='--worker' and args[1] in ('frozen-RAW10-s2026092701','frozen-ANTI-s2026092701')):raise ValueError('No diagnostic data/views/stream/model/parameter overrides')
 d=bootstrap()
 if args:
  from experiments.cvs_f_postas_noise_probe.probe import run
  row=next(r for r in d.rows() if r['row_id']==args[1])
  try:run(row,'cuda:0')
  except Exception as e:d.write(d.BASE/(row['row_id']+'.failure.json'),dict(status='FAILED',error=repr(e),no_automatic_retry=True,preserve_outputs=True));raise
 else:
  from experiments.cvs_f_postas_noise_probe.dispatch import run
  run()
