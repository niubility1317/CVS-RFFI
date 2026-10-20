import copy
from pathlib import Path
from types import SimpleNamespace
import pytest
from experiments.pamf_r38_successor import design as d,runtime as r,evaluation as e,runner


def trained(arm):
    return dict(status='SOURCE_TRAINED',arm=arm,epochs=10,updates=2470,L_exposures=63000,U_exposures=567000,parameters=d.PARAMETERS[arm],
        initialization_IQ_examples=0,teacher_parameter_updates=0,context_updates=2470,prior_updates=2470,cumulative_prior_epochs=50,
        cumulative_coarse_epochs=50,cumulative_action_epochs=50,TX_labels_read=0,V_reads=0,target_reads=0,training_F_access=d.TASK[arm],
        F_weights_read=int(d.TASK[arm]),F_outputs_read=1260000*int(d.TASK[arm]),F_parameter_updates=0,head_fit_updates=0,paired_clean_optimized=True)


@pytest.mark.parametrize('bad',['root','role','arm','kind','plan'])
def test_invalid_target_reference_rejects_before_arrays(monkeypatch,bad):
    args=SimpleNamespace(output=Path('/runs')/d.RUN/d.ARMS[0],kind=d.KIND);role='target'
    if bad=='root':args.output=Path('/runs/unknown')/d.ARMS[0]
    elif bad=='role':role='query'
    elif bad=='arm':args.output=args.output.parent/'unknown'
    elif bad=='kind':args.kind='other'
    plan=copy.deepcopy(d.configuration());plan['updates']=2469
    monkeypatch.setattr(e,'read_json',lambda *a:plan)
    monkeypatch.setattr(e.np,'load',lambda *a,**k:pytest.fail('Arrays stay closed'))
    with pytest.raises(ValueError):e.reference(args,role)


@pytest.mark.parametrize('field,bad',[('epochs',20),('updates',4940),('cumulative_prior_epochs',60),('F_weights_read',1),('TX_labels_read',1),('V_reads',1),('target_reads',1)])
def test_wrong_ten_epoch_source_budget_or_F_free_control_stops_target(monkeypatch,field,bad):
    row=Path('/runs')/d.RUN/'half_iq';value=trained('half_iq');value[field]=bad
    monkeypatch.setattr(e,'read_json',lambda p:value if p.name=='training_complete.json' else dict(status='ANALYZED'))
    with pytest.raises(ValueError):e.validate_source_row(row)


def test_parent_source_contract_check_never_opens_scores_truth_or_IQ(monkeypatch):
    calls=[]
    def read(path):
        calls.append(path.name)
        if path.name=='launch.json':return dict(commit=d.COMPACT_PARENT_COMMIT)
        if path.name=='comparison_plan.json':return d.parent.configuration()
        assert path.name=='training_complete.json'
        return dict(status='SOURCE_TRAINED',arm='dense_half',epochs=20,updates=4940,L_exposures=126000,U_exposures=1134000,
            parameters=794529,TX_labels_read=0,F_weights_read=0,F_outputs_read=0,training_F_access=False,V_reads=0,target_reads=0,
            cumulative_prior_epochs=40,cumulative_coarse_epochs=40,cumulative_action_epochs=40)
    monkeypatch.setattr(r,'read_json',read);r.validate_parent_root(Path('/runs')/d.COMPACT_PARENT_RUN)
    assert calls==['launch.json','comparison_plan.json','training_complete.json']


@pytest.mark.parametrize('arm',('half_iq','half_task'))
def test_each_row_automatically_runs_full_source_and_target_without_score_gate(tmp_path,monkeypatch,arm):
    args=SimpleNamespace(output=tmp_path,arm=arm);(tmp_path/arm).mkdir();seen=[]
    monkeypatch.setattr(runner,'train',lambda *a:seen.append('train'))
    monkeypatch.setattr(runner,'predict',lambda a,role:seen.append('predict_'+role))
    monkeypatch.setattr(runner,'profile',lambda *a:seen.append('profile'))
    monkeypatch.setattr(runner,'score_subprocess',lambda a,role:seen.append('score_'+role))
    monkeypatch.setattr(runner.torch.cuda,'empty_cache',lambda:None)
    runner.worker(args)
    assert seen==['train','predict_source','profile','score_source','predict_target','score_target']
    assert (tmp_path/arm/'complete.json').exists()


@pytest.mark.parametrize('arm',('half_iq','half_task'))
def test_independent_CPU_truth_side_waits_for_all_predictions(monkeypatch,arm):
    args=SimpleNamespace(output=Path('/runs')/d.RUN,arm=arm)
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','')
    monkeypatch.setattr(e,'reference',lambda *a:(Path('/formal'),e.np.arange(2),None,None))
    class Missing(Exception):pass
    def incomplete(*a):raise Missing()
    monkeypatch.setattr(e.base,'prediction_arrays',incomplete)
    monkeypatch.setattr(e.base,'target_baseline',lambda *a:pytest.fail('No truth/baseline before predictions'))
    with pytest.raises(Missing):e.score(args,'target')


def test_capacity_queue_requires_old_row_terminal_and_actual_owned_handle_exit(tmp_path,monkeypatch):
    import json
    arm='keep_iq';previous=tmp_path/d.PREDECESSOR_BY_GPU[d.GPUS[arm]];previous.mkdir();(previous/'old').mkdir()
    (previous/'workers.json').write_text(json.dumps(dict(workers=[dict(arm='old',pid=123,GPU=d.GPUS[arm])])),encoding='utf-8')
    args=SimpleNamespace(output=tmp_path/d.RUN)
    assert not runner.predecessor_ready(args,arm)
    (previous/'old/complete.json').write_text('{}',encoding='utf-8')
    class Proc:
        state='S';owned=True
        def __truediv__(self,key):self.key=key;return self
        def read_text(self):return '123 (python) '+self.state+' unused'
        def read_bytes(self):
            values=[str(previous),d.PREDECESSOR_MODULE[previous.name]] if self.owned else ['unrelated']
            return chr(0).join(values).encode()
    proc=Proc();monkeypatch.setattr(runner,'Path',lambda value:proc if value=='/proc' else Path(value))
    assert not runner.predecessor_ready(args,arm)
    proc.state='Z';assert runner.predecessor_ready(args,arm)
    proc.state='S';proc.owned=False;assert runner.predecessor_ready(args,arm)


def test_technical_failure_does_not_dispatch_target_or_retry(tmp_path,monkeypatch):
    args=SimpleNamespace(output=tmp_path,arm=d.ARMS[0])
    def fail(*a):raise FloatingPointError('nonfinite')
    monkeypatch.setattr(runner,'train',fail);monkeypatch.setattr(runner,'predict',lambda *a:pytest.fail('No target'))
    with pytest.raises(FloatingPointError):runner.worker(args)
    assert (tmp_path/args.arm/'failure.json').exists()


def test_actual_successor_prediction_entry_accepts_E10_before_any_reader_or_array_IO(tmp_path,monkeypatch):
    args=SimpleNamespace(output=tmp_path/d.RUN,arm='half_iq',reader_folder=Path('/reader'),source_cache=Path('/source'))
    class Model:
        def requires_grad_(self,value):assert value is False;return self
    class ReachedReader(Exception):pass
    monkeypatch.setattr(e,'reference',lambda *a:(Path('/received'),e.np.arange(2),None,None))
    monkeypatch.setattr(e,'load_checkpoint',lambda *a:(Model(),dict(arm='half_iq',epoch=10,updates=2470,kind=d.KIND)))
    def reader(*a,**k):raise ReachedReader()
    monkeypatch.setattr(e.base,'load_reader',reader)
    with pytest.raises(ReachedReader):e.predict(args,'source')


@pytest.mark.parametrize('budget,accepted',[((20,4940),True),((10,2470),False)])
def test_shared_prediction_default_still_requires_old_E20_final(monkeypatch,budget,accepted):
    args=SimpleNamespace(output=Path('/unused'),kind=d.KIND,reader_folder=Path('/reader'),source_cache=Path('/source'))
    class Model:
        def requires_grad_(self,value):return self
    class ReachedReader(Exception):pass
    def reader(*a,**k):raise ReachedReader()
    monkeypatch.setattr(e.base,'load_reader',reader)
    loader=lambda *a:(Model(),dict(epoch=budget[0],updates=budget[1],kind=d.KIND))
    reference=lambda *a:(Path('/received'),e.np.arange(2),None,None)
    with pytest.raises(ReachedReader if accepted else ValueError):e.base.predict(args,'source',checkpoint_loader=loader,reference_loader=reference)
