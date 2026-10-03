import json
from pathlib import Path

import numpy as np
import pytest

from experiments.cvs_frontfilter_attribution.collect import CONDITIONS, SOURCE_COMMIT, recount_row


def write(p,v):p.write_text(json.dumps(v),encoding='utf-8')
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def change(p,fn):
    d=read(p);fn(d);write(p,d)
def change_npz(p,fn):
    with np.load(p,allow_pickle=False) as z:d={k:z[k] for k in z.files}
    fn(d);np.savez(p,**d)


@pytest.fixture
def result(tmp_path):
    source=tmp_path/'source';source.mkdir()
    out=tmp_path/'out';out.mkdir();(out/'source_loader').mkdir()
    lids=['L'+str(i) for i in range(6300)];vids=['V'+str(i) for i in range(27000)]
    contract=dict(role_ids=dict(L_s=lids,V=vids))
    write(source/'source_contract.json',contract);write(out/'source_loader/source_contract.json',contract)
    variant='frontfilter_dynamic';seed=2026092701
    write(out/'resolved_config.json',dict(source_output=str(source),source_commit=SOURCE_COMMIT,variant=variant,model_seed=seed,source_record=dict(accuracy=1.,worst_rx=1.)))
    coeff=np.zeros((6300,2,4),dtype=np.float32);coeff[:,0,0]=.5
    np.savez(out/'source_L_coefficients.npz',ids=np.asarray(lids),coefficients=coeff)
    write(out/'mean_L_coefficients.json',dict(role='L_s',count=6300,used_labels=False,frozen_before_V=True,ids=lids,mean=coeff.astype(np.float64).mean(0).astype(np.float32).tolist()))
    grid=np.asarray([(t,r,d) for t in range(6) for r in (1,3,4,6,8) for d in (1,2,3) for _ in range(300)])
    truth,rx,day=grid.T;rows=[]
    for condition in CONDITIONS:
        logits=np.zeros((27000,6),dtype=np.float32);logits[np.arange(27000),truth]=2.
        if condition!='all_on':logits[0,1]=3.
        pred=logits.argmax(1);correct=pred==truth
        shifted=logits.astype(np.float64)-logits.max(1,keepdims=True)
        ce=float((np.log(np.exp(shifted).sum(1))-shifted[np.arange(27000),truth]).mean())
        rates={str(r):float(correct[rx==r].mean()) for r in (1,3,4,6,8)}
        rows.append(dict(condition=condition,accuracy=float(correct.mean()),ce=ce,rx_accuracy=rates,worst_rx=min(rates.values()),prediction_changes=int((~correct).sum()),helped_by_all_on=int((~correct).sum()),hurt_by_all_on=0))
        np.savez(out/(condition+'_source_predictions.npz'),ids=np.asarray(vids),truth=truth,receiver=rx,day=day,logits=logits,predictions=pred)
    values=np.arange(27000,dtype=np.float64)/27000
    stats=dict(mean=float(values.mean()),p10=float(np.quantile(values,.1)),median=float(np.median(values)),p90=float(np.quantile(values,.9)),maximum=float(values.max()))
    np.savez(out/'source_scalar_diagnostics.npz',ids=np.asarray(vids),receiver=rx,day=day,input_change=values)
    write(out/'completion.json',dict(variant=variant,model_seed=seed,conditions=list(CONDITIONS),coefficient_role='L_s',coefficient_count=6300,target_access=False,optimizer_updates=0,source_selection_changed=False,model_unchanged=True,rows=rows,scalar_summary=dict(input_change=stats)))
    return out


def test_recounts_L_mean_predictions_CE_and_all90_cells(result):
    d=recount_row(result)
    assert d['coefficient_recount'] and d['coefficient_mean']==[[.5,0.,0.,0.],[0.,0.,0.,0.]]
    assert len(d['records'])==3 and all(len(r['cells'])==90 for r in d['records'])
    assert d['records'][1]['prediction_changes']==1 and d['records'][1]['helped_by_all_on']==1
    assert d['records'][1]['contribution_pp']==pytest.approx(100/27000)
    assert sum(map(sum,d['records'][1]['confusions']['ALL']))==27000
    assert d['records'][1]['macro_f1']<1


@pytest.mark.parametrize('field,value',[
    ('role','V'),('count',6299),('used_labels',True),('frozen_before_V',False),('mean',[[0.,0.,0.,0.],[0.,0.,0.,0.]])])
def test_rejects_invalid_mean_summary(result,field,value):
    change(result/'mean_L_coefficients.json',lambda d:d.update({field:value}))
    with pytest.raises(ValueError):recount_row(result)


@pytest.mark.parametrize('field,value',[('target_access',True),('optimizer_updates',1),('source_selection_changed',True),('model_unchanged',False),('model_seed',12),('conditions',['all_on','g_identity'])])
def test_rejects_changed_diagnostic_scope(result,field,value):
    change(result/'completion.json',lambda d:d.update({field:value}))
    with pytest.raises(ValueError):recount_row(result)


def test_rejects_wrong_CE_even_when_argmax_matches(result):
    change(result/'completion.json',lambda d:d['rows'][1].update(ce=9.))
    with pytest.raises(ValueError,match='CE'):recount_row(result)


def test_rejects_predicted_label_not_logit_argmax(result):
    change_npz(result/'mean_L_source_predictions.npz',lambda d:d['predictions'].__setitem__(0,0))
    with pytest.raises(ValueError,match='logits'):recount_row(result)


def test_rejects_nonfinite_L_coefficients(result):
    change_npz(result/'source_L_coefficients.npz',lambda d:d['coefficients'].__setitem__((0,0,0),np.nan))
    with pytest.raises(ValueError):recount_row(result)


def test_rejects_validation_id_in_L_coefficients(result):
    change_npz(result/'source_L_coefficients.npz',lambda d:d['ids'].__setitem__(0,'V0'))
    with pytest.raises(ValueError,match='L_s'):recount_row(result)


def test_rejects_changed_physical_V_pairing(result):
    change_npz(result/'mean_L_source_predictions.npz',lambda d:d['ids'].__setitem__(0,'V1'))
    with pytest.raises(ValueError):recount_row(result)


def test_rejects_changed_day_pairing(result):
    change_npz(result/'mean_L_source_predictions.npz',lambda d:d['day'].__setitem__(0,2))
    with pytest.raises(ValueError,match='Unpaired'):recount_row(result)


def test_rejects_missing_class_cell(result):
    for condition in CONDITIONS:
        change_npz(result/(condition+'_source_predictions.npz'),lambda d:d['day'].__setitem__(0,2))
    with pytest.raises(ValueError,match='Unbalanced'):recount_row(result)


def test_rejects_changed_scalar_values(result):
    change_npz(result/'source_scalar_diagnostics.npz',lambda d:d['input_change'].__setitem__(0,2.))
    with pytest.raises(ValueError,match='Scalar'):recount_row(result)


def test_rejects_changed_source_commit(result):
    change(result/'resolved_config.json',lambda d:d.update(source_commit='other'))
    with pytest.raises(ValueError):recount_row(result)
