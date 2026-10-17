"""Source-only statistic or synthetic checks;no target/runtime model import."""
import json,sys
from pathlib import Path
here=Path(__file__).resolve().parents[2];sys.path.insert(0,str(here))
if __name__=='__main__':
 if len(sys.argv)==2 and sys.argv[1]=='--checks':
  from experiments.cvs_envelope_profile17.checks import run
  (here/'checks.json').write_text(json.dumps(run(),indent=2)+'\n')
 elif len(sys.argv)==1:
  from experiments.cvs_envelope_profile17.worker import run
  run()
 else:raise ValueError('Only fixed source statistic or synthetic checks')
