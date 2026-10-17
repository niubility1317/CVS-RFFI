from pathlib import Path

def loader_kind(path,baseline,final):
 p=Path(path)
 if baseline and p==Path(baseline):return 'parent'
 if p==Path(final):return 'final'
 raise ValueError('Unexpected external checkpoint dependency')

def supervision_weight(mode,epoch,qualified):
 if mode not in ('B0','URAW','UAS') or epoch not in range(14,31):raise ValueError('Fixed U mode/epoch only')
 return 0. if mode=='B0' or not qualified else .16*min(1.,(epoch-13)/5.)
