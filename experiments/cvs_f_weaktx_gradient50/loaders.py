"""Reuse exact original reference release; never initialize new training from it."""
from pathlib import Path
import importlib.util,json,sys
from core import REF_RUN,REF_COMMIT,reference_scope
ROOT=Path(__file__).resolve().parents[2]
SEEDS=(2026092701,2026092702,2026092703)
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def imports(a):
 origin=Path(a.source_release).resolve();source=Path(a.source_run).resolve();profile=Path(a.profile).resolve();reference_scope(source.name,(origin/'release_commit.txt').read_text(encoding='utf-8').strip());sys.path[1:1]=[str(origin),str(origin/'code')]
 from experiments.cvs_f_crossrx50_local import design as d,model as api
 from experiments.cvs_f_crossrx50_local.provenance import source_proof,validate_checkpoint
 from experiments.cvs_f_supcon50.mechanism import penalty
 from experiments.cvs_equivariant_identity.precision import numerical_context
 if d.ROOT.resolve()!=origin or d.RUN!=REF_RUN:raise ValueError('Wrong actual baseline implementation')
 configs=[]
 for seed in SEEDS:
  c=read(source/('BASE-s'+str(seed))/'source/local_source_complete.json')['config'];source_proof(c)
  if c['treatment']!='BASE' or c['model_seed']!=seed or Path(c['runtime_profile']).resolve()!=profile or Path(c['runtime_base']).resolve()!=source:raise ValueError('Actual reference source/profile differs')
  configs.append(c)
 spec=importlib.util.spec_from_file_location('diagnostic_tail_formula',ROOT/'experiments/cvs_f_weaktx_tailrisk50/loss.py');tail=importlib.util.module_from_spec(spec);spec.loader.exec_module(tail)
 return d,api,validate_checkpoint,penalty,tail,numerical_context,configs
