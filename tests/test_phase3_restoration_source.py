import numpy as np
import pytest
from experiments.pamf_phase3_restoration.source_pair import selected_indices,validate_source,EPISODE,SOURCE_FIELDS


def test_fixed_source_subset_is_rolewise_linspace_with_all_endpoints():
    for role,n,k in [('L_s',6300,128),('U_s',56700,384)]:
        actual=selected_indices(role)
        assert np.array_equal(actual,np.linspace(0,n-1,k,dtype=int))
        assert len(set(actual))==k and actual[0]==0 and actual[-1]==n-1
    for role in ['V','OE','novel','target','Q']:
        with pytest.raises(ValueError,match='only allow'):
            selected_indices(role)


@pytest.mark.parametrize('extra',['TX_labels','OE_root','target_U','target_Q','source_V','roles','raw_clean_target'])
def test_source_boundary_rejects_training_role_expansion_before_any_file_or_import(extra):
    value={key:'original' for key in SOURCE_FIELDS};value['episode']=EPISODE
    value[extra]='forbidden'
    with pytest.raises(ValueError,match='Only original'):
        validate_source(value)


@pytest.mark.parametrize('full',[False,True])
def test_source_locator_metadata_never_fetches_L_labels_or_any_other_role(monkeypatch,full):
    import sys
    from types import SimpleNamespace
    from experiments.pamf_phase3_restoration import source_pair as m
    counts=m.COUNTS
    ids={r:np.asarray([r+':'+str(i) for i in range(counts[r])]) for r in ('L_s','U_s')}
    contract={'counts':counts,'role_ids':{r:a.tolist() for r,a in ids.items()}}
    module=SimpleNamespace(_reader_config=lambda c:None,_contract=lambda c:contract)
    monkeypatch.setitem(sys.modules,'experiments.pamf_phase3_open_world.star_relative_source',module)
    accessed=[]
    class Store:
        def __init__(self,role):self.role=role;self.files=list(m.METADATA)+(['y'] if role=='L_s' else [])
        def __enter__(self):return self
        def __exit__(self,*args):return False
        def __getitem__(self,key):
            accessed.append((self.role,key))
            if key=='y':raise AssertionError('Direct TX label read forbidden')
            return ids[self.role] if key=='ids' else np.zeros(counts[self.role],np.int64)
    def load(path,**kwargs):
        name=path.name
        assert name in ['L_s_metadata.npz','U_s_metadata.npz']
        return Store('L_s' if name.startswith('L_s') else 'U_s')
    monkeypatch.setattr(m.np,'load',load)
    c={k:'source' for k in SOURCE_FIELDS};c['episode']=EPISODE
    _,meta=m.metadata(c,full=full)
    assert len(meta['ids'])==(63000 if full else 512) and 'y' not in meta
    if full:
        assert np.array_equal(meta['ids'],np.concatenate([ids['L_s'],ids['U_s']]))
    assert {r for r,k in accessed}=={'L_s','U_s'} and all(k!='y' for r,k in accessed)
