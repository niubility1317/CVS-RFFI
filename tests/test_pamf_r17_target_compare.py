import copy
from pathlib import Path
from types import SimpleNamespace
import pytest

from experiments.pamf_r17_target_compare import design as d,evaluation as e
from experiments.pamf_uncertainty_restore import evaluation as original
from experiments.pamf_r17_task import evaluation as task


def test_fixed_plan_contains_both_frozen_models_without_forged_source_qualification():
    plan=d.configuration()
    assert plan['rows']==['step_norm_task','activation_task'] and plan['count_per_view']==168000
    assert plan['target_acceptance_allowance_pp']==1 and plan['new_training_updates']==0
    assert 'notforged' in plan['source_result'] and plan['target_feedback'] is False
    assert d.SOURCE_COMMIT=='a310b3b5ff30e23584e7bc950e8a0ff96ac685fb'


@pytest.mark.parametrize('bad',['role','row','root','kind'])
def test_wrong_role_or_row_rejected_before_any_IO(monkeypatch,bad):
    args=SimpleNamespace(output=Path('/runs')/d.RUN/d.ARMS[0],kind=d.KIND)
    role='target'
    if bad=='role':role='source'
    elif bad=='row':args.output=args.output.parent/'unknown'
    elif bad=='root':args.output=Path('/runs/other')/d.ARMS[0]
    else:args.kind='other'
    monkeypatch.setattr(e,'read_json',lambda *a:pytest.fail('No IO for unregistered row'))
    with pytest.raises(ValueError):e.target_reference(args,role)


@pytest.mark.parametrize('bad',['count','rows','source_commit'])
def test_changed_plan_keeps_target_arrays_closed(monkeypatch,bad):
    plan=copy.deepcopy(d.configuration())
    if bad=='count':plan['count_per_view']=1
    elif bad=='rows':plan['rows'].pop()
    else:plan['source_commit']='unknown'
    args=SimpleNamespace(output=Path('/runs')/d.RUN/d.ARMS[0],kind=d.KIND)
    monkeypatch.setattr(e,'read_json',lambda *a:plan)
    monkeypatch.setattr(e.np,'load',lambda *a,**k:pytest.fail('Target IQ/IDs/truth remain closed'))
    with pytest.raises(ValueError,match='comparison'):e.target_reference(args,'target')


@pytest.mark.parametrize('key,value',[('training_F_access',False),('V_reads',1),('target_reads',1),('updates',4939),('F_parameter_updates',1)])
def test_source_budget_or_permissions_mismatch_rejected(monkeypatch,key,value):
    root=Path('/runs')/d.SOURCE_RUN
    def read(path):
        if path.name=='launch.json':return dict(commit=d.SOURCE_COMMIT)
        if path==root/'complete.json':return dict(status='ANALYZED',source_rows=list(d.ARMS),target_rows=[])
        if path.name=='complete.json':return dict(status='ANALYZED')
        arm=path.parent.name
        data=dict(status='SOURCE_TRAINED',arm=arm,epochs=20,updates=4940,L_exposures=126000,U_exposures=1134000,
            teacher_parameter_updates=0,frozen_context_updates=0,TX_labels_read=0,V_reads=0,target_reads=0,
            training_F_access=True,F_parameter_updates=0,head_fit_updates=0,parameters=d.PARAMETERS[arm])
        data[key]=value
        return data
    monkeypatch.setattr(e,'read_json',read)
    with pytest.raises(ValueError,match='source budget'):e.validate_source(root)


def test_legacy_prediction_and_scoring_keep_their_existing_reference_by_default(monkeypatch):
    class Expected(Exception):pass
    def reject(*args):raise Expected('Original qualification still enforced')
    monkeypatch.setattr(original,'reference',reject)
    with pytest.raises(Expected):original.predict(SimpleNamespace(),'target')
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','')
    with pytest.raises(Expected):task.score_original_head(SimpleNamespace(),'target')


def test_new_prediction_uses_explicit_comparison_reference_and_actual_frozen_final(monkeypatch):
    args=SimpleNamespace(output=Path('/runs')/d.RUN,arm=d.ARMS[0],source_run=Path('/runs')/d.SOURCE_RUN)
    monkeypatch.setattr(e,'validate_source',lambda *a:None)
    seen=[]
    def load(path,device):
        seen.append((path,device))
        return object(),dict(arm=args.arm,epoch=20,updates=4940)
    monkeypatch.setattr(e,'load_checkpoint',load)
    def forward(row,role,**kwargs):
        assert role=='target' and kwargs['reference_loader'] is e.target_reference
        model,payload=kwargs['checkpoint_loader'](row.output/d.KIND/'final.pt','cpu')
        assert payload['epoch']==20
        with pytest.raises(ValueError):kwargs['checkpoint_loader'](row.output/'different.pt','cpu')
    monkeypatch.setattr(e,'original_predict',forward)
    e.predict(args)
    assert seen==[(args.source_run/args.arm/d.KIND/'final.pt','cpu')]


def test_new_reference_opens_only_existing_formal_IDs_after_plan_and_manifest(monkeypatch):
    args=SimpleNamespace(output=Path('/runs')/d.RUN/d.ARMS[0],kind=d.KIND,formal_reference=Path('/formal/reference'))
    calls=[]
    def read(path):
        calls.append(path.name)
        return d.configuration() if path.name=='comparison_plan.json' else dict(status='FORMAL_REFERENCE_READY',count=168000,
            views=list(d.VIEWS),original_IQ_bytes_reused=True,TX_truth_reads=0)
    monkeypatch.setattr(e,'read_json',read)
    def load(path,**kwargs):
        assert calls==['comparison_plan.json','manifest.json'] and path.name=='physical_ids.npy'
        return e.np.arange(168000)
    monkeypatch.setattr(e.np,'load',load)
    folder,ids,source,index=e.target_reference(args,'target')
    assert len(ids)==168000 and source is index is None and folder==args.formal_reference
