"""Check source simulation activation and F-free selection boundaries."""
import ast
from pathlib import Path

source = Path(__file__).with_name('phase1_complex_multires_source.py').read_text(encoding='utf8')
tree = ast.parse(source)
assert not any(isinstance(n, ast.ImportFrom) and (n.module or '').startswith('leo_practical') for n in tree.body), 'channel imports must follow activation'
fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'source_inputs')
activations = [n.lineno for n in ast.walk(fn) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == 'activate']
imports = [n.lineno for n in ast.walk(fn) if isinstance(n, ast.ImportFrom) and (n.module or '').startswith('leo_practical')]
assert activations and imports and max(activations) < min(imports), 'activate immutable source runtime before channel imports'
main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
main_source = ast.get_source_segment(source, main)
assert main_source.index("args.out/'source_selection.json'") < main_source.index('core.teacher('), 'source selection must precede loading F'
print('VERIFIED source dependency activation and F-free selection boundary')
