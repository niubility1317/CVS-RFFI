"""Focused source-V scorer and leakage-boundary checks."""
import ast
from pathlib import Path
import numpy as np

path=Path(__file__).with_name('phase1_source_v_restoration_diagnostic.py')
source=path.read_text(encoding='utf8');tree=ast.parse(source)
fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='score_counts')
ns={};exec(compile(ast.Module(body=[fn],type_ignores=[]),'<counts-test>','exec'),ns)
y=np.array([0,1,2,3,4]);base=np.array([0,0,2,1,4]);candidate=np.array([0,1,1,1,4])
assert ns['score_counts'](y,base,candidate)==dict(count=5,baseline_correct=3,candidate_correct=3,repairs=1,damages=1)
predict=next(n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name=='predict')
text=ast.get_source_segment(source,predict)
assert "['y']" not in text and "['label']" not in text
score=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='score')
text=ast.get_source_segment(source,score)
assert text.index("arr=np.load")<text.index("truth=z['y']")
print('VERIFIED paired counts and source V truth-last boundary')
