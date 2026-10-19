from pathlib import Path

def loader_kind(path,baseline,final):
 p=Path(path)
 if baseline and p==Path(baseline):return 'parent'
 if p==Path(final):return 'final'
 raise ValueError('Unexpected external checkpoint dependency')
