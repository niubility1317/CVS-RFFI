"""Import the historical factory together with its repository dependencies."""
import importlib.util
from pathlib import Path
import sys

def load_factory(release):
 release=Path(release)
 if str(release) not in sys.path:sys.path.insert(0,str(release))
 spec=importlib.util.spec_from_file_location('_sameIQ_CSIL_native_models',release/'comparison_suite/models.py')
 api=importlib.util.module_from_spec(spec);spec.loader.exec_module(api)
 return api

def import_regression():
 import tempfile
 with tempfile.TemporaryDirectory() as tmp:
  root=Path(tmp);(root/'comparison_suite').mkdir();(root/'fixture_teacher_dependency.py').write_text('VALUE=6\n')
  path=root/'comparison_suite/models.py';path.write_text('from fixture_teacher_dependency import VALUE\n')
  oldpath=list(sys.path)
  try:
   spec=importlib.util.spec_from_file_location('_old_isolated_teacher',path);api=importlib.util.module_from_spec(spec)
   try:spec.loader.exec_module(api)
   except ModuleNotFoundError:pass
   else:raise AssertionError('Isolated historical import did not reproduce failure')
   if load_factory(root).VALUE!=6:raise AssertionError('Historical factory dependency route failed')
  finally:
   sys.path[:]=oldpath;sys.modules.pop('fixture_teacher_dependency',None)
 return dict(status='PASS',isolated_old_import_failed=True,repository_dependency_import_passed=True)
