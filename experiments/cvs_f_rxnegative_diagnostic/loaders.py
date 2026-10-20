"""Use fixed original local BASE release; no checkpoint becomes training state."""
from pathlib import Path
import json,sys
from core import REF_RUN,REF_COMMIT,SEEDS
ROOT=Path(__file__).resolve().parents[2]
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def imports(a):
 origin=Path(a.source_release).resolve();source=Path(a.source_run).resolve();profile=Path(a.profile).resolve()
 if source.name!=REF_RUN or (origin/'release_commit.txt').read_text().strip()!=REF_COMMIT:raise ValueError('Exact completed original local F BASE release only')
 sys.path[1:1]=[str(origin),str(origin/'code')]
 from experiments.cvs_f_crossrx50_local import design as d,model as api
 from experiments.cvs_f_crossrx50_local.provenance import source_proof,validate_checkpoint
 from experiments.cvs_equivariant_identity.precision import numerical_context
 if d.ROOT.resolve()!=origin or d.RUN!=REF_RUN:raise ValueError('Original immutable BASE implementation required')
 configs=[]
 for seed in SEEDS:
  s=read(source/('BASE-s'+str(seed))/'source/local_source_complete.json');c=s['config'];source_proof(c)
  if s['source_commit']!=REF_COMMIT or c['treatment']!='BASE' or c['model_seed']!=seed or Path(c['runtime_profile']).resolve()!=profile or Path(c['runtime_base']).resolve()!=source:raise ValueError('Source data/profile/original matching baseline differs')
  configs.append(c)
 return d,api,validate_checkpoint,numerical_context,configs
