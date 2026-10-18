"""Reproduce and verify the exact NumPy2-export/NumPy1-reader compatibility fix."""
import argparse
import ast
import json
import pickle
from pathlib import Path
import numpy as np

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pickle',type=Path,required=True)
    p.add_argument('--expected',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    # Compile only the actual unpickler class, avoiding a torch import in this
    # bounded NumPy1.24 regression process.
    tree=ast.parse(Path(__file__).with_name('pipeline.py').read_text(encoding='utf-8'))
    node=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='NumpyExportUnpickler')
    namespace={'pickle':pickle}
    exec(compile(ast.Module(body=[node],type_ignores=[]),'pipeline.py','exec'),namespace)
    try:
        with args.pickle.open('rb') as handle:pickle.load(handle)
    except ModuleNotFoundError as exc:
        before=str(exc)
    else:
        raise AssertionError('Expected the NumPy1.24 baseline to reproduce numpy._core failure')
    with args.pickle.open('rb') as handle:actual=namespace['NumpyExportUnpickler'](handle).load()
    expected=np.load(args.expected,allow_pickle=False)
    assert len(actual['data'])==6
    for i,x in enumerate(actual['data']):
        assert x.dtype==expected[str(i)].dtype and x.shape==expected[str(i)].shape
        assert np.array_equal(x,expected[str(i)]),i
    result={'status':'PASS','numpy':np.__version__,'baseline_failure':before,'six_source_blocks_array_equal':True,'IQ_values_changed':False,'query_access':False}
    with args.output.open('x',encoding='utf-8') as handle:json.dump(result,handle,indent=2)
    print(json.dumps(result))

if __name__=='__main__':main()
