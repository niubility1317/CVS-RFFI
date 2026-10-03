import copy,itertools
import numpy as np
import pytest
from experiments.cvs_validdual_mechanism.analyze import AXES,decompose,public_signals,received_views
from experiments.cvs_validdual_mechanism.recount import contrast_energies

def cells(x,variance=.2):
    return [dict(tx=AXES[0][i],receiver=AXES[1][j],day=AXES[2][k],count=300,coefficient_mean=x[i,j,k].tolist(),coefficient_trace_variance=variance) for i,j,k in itertools.product(range(6),range(5),range(3))]

def test_balanced_decomposition_independent_contrasts():
    x=np.random.default_rng(9).normal(size=(6,5,3,66));d=decompose(cells(x));ref=contrast_energies(x)
    for name,value in ref.items():assert d['components'][name]==pytest.approx(value,rel=1e-12)
    assert sum(d['fractions_of_total'].values())+d['within_fraction']==pytest.approx(1)

@pytest.mark.parametrize('axis,name',[(0,'TX'),(1,'RX'),(2,'day')])
def test_isolated_factor(axis,name):
    shape=[1,1,1,1];shape[axis]=len(AXES[axis]);x=np.broadcast_to(np.arange(len(AXES[axis])).reshape(shape),(6,5,3,66)).copy()
    d=decompose(cells(x,0));assert d['fractions_of_total'][name]==pytest.approx(1)
    assert all(v<1e-20 for k,v in d['components'].items() if k!=name)

@pytest.mark.parametrize('bad',['missing','duplicate','rx','unbalanced','nonfinite','negative'])
def test_bad_moments_rejected(bad):
    c=cells(np.zeros((6,5,3,66)))
    if bad=='missing':c.pop()
    if bad=='duplicate':c[-1]=copy.deepcopy(c[0])
    if bad=='rx':c[0]['receiver']=0
    if bad=='unbalanced':c[0]['count']=299
    if bad=='nonfinite':c[0]['coefficient_mean'][0]=float('nan')
    if bad=='negative':c[0]['coefficient_trace_variance']=-.1
    with pytest.raises(ValueError):decompose(c)

def test_constant_static_not_spurious_percentages():
    d=decompose(cells(np.ones((6,5,3,66)),1e-16));assert d['degenerate'] and d['within_fraction'] is None
    assert set(d['fractions_of_total'].values())=={None}

def test_public_history_and_rms():
    records,z=public_signals();views=received_views(z);assert len(records)==16 and z.shape==(16,320)
    y=z[:,64:]+(.23+.09j)*z[:,48:304];y/=np.sqrt(np.mean(abs(y)**2,axis=1,keepdims=True))
    np.testing.assert_array_equal(views['delay16'],np.stack((y.real,y.imag),1).astype(np.float32))
    assert all(np.isfinite(v).all() for v in views.values())
    for v in views.values():np.testing.assert_allclose((v*v).sum(1).mean(1),1,atol=2e-7)

@pytest.mark.parametrize('tamper',[None,'missing_last','wrong_metric','within_fraction','cell_count','degenerate','model_label','public_label'])
def test_full_recount_and_negative_cases(tmp_path,monkeypatch,tamper):
    import torch
    from experiments.cvs_validdual_mechanism import analyze as a,recount as r
    from experiments.cvs_validdual_clean.contracts import expected_rows
    class Toy(torch.nn.Module):
        def parts(self,x):
            raw=x[:,:,:4].flatten(1);z=raw*2;g=x*.8;c=x[:,:,:33]*.1
            return z,raw,z,g,c
    out=tmp_path/'analysis';out.mkdir();source=tmp_path/'source';source.mkdir()
    monkeypatch.setattr(r,'OUTPUT',tmp_path);monkeypatch.setattr(r,'SOURCE',source)
    records,waves=a.public_signals();views=a.received_views(waves)
    np.savez_compressed(out/'public_inputs.npz',continuous=waves,**views)
    a.write(out/'public_records.json',records)
    allf=[];allm=[]
    for rid in expected_rows():
        folder=source/rid;folder.mkdir();cc=cells(np.random.default_rng(123).normal(size=(6,5,3,66)))
        stored=copy.deepcopy(cc)
        if tamper=='cell_count':stored[0]['count']=1
        a.write(folder/'source_final_diagnostics.json',dict(validdual_groups=stored))
        f=a.decompose(cc);means=f.pop('cell_means');f['row_id']=rid;allf.append(f)
        variant,seed=expected_rows()[rid];f.update(variant=variant,model_seed=seed)
        arrays,metrics=a.probe(Toy(),views);np.savez_compressed(out/(rid+'.npz'),cell_means=means,**arrays)
        for m in metrics:m.update(row_id=rid,variant=variant,model_seed=seed,**records[m['public_index']]);allm.append(m)
    if tamper=='missing_last':allm.pop()
    if tamper=='wrong_metric':allm[-1]['fused_feature_drift']+=.1
    if tamper=='within_fraction':allf[-1]['within_fraction']=7.
    if tamper=='degenerate':allf[-1]['degenerate']=True
    if tamper=='model_label':allm[-1]['model_seed']=42
    if tamper=='public_label':allm[-1]['family']='noise'
    a.write(out/'complete.json',dict(status='ARTIFACTS_COMPLETE',rows=8,source_cells=720,public_pairs=512,target_access=False,new_checkpoints=False))
    a.write(out/'source_factors.json',dict(rows=allf));a.write(out/'public_metrics.json',dict(rows=allm))
    if tamper:
        with pytest.raises(ValueError):r.recount()
        assert not (out/'independent_recount.json').exists()
    else:
        r.recount();assert a.read(out/'independent_recount.json')['status']=='VERIFIED'
        with pytest.raises(FileExistsError):r.recount()
