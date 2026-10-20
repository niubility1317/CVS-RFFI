import json,sys
from types import SimpleNamespace
import numpy as np
import pytest
from experiments.pamf_oracle_attribution import design as d,runner
from experiments.pamf_oracle_attribution.operators import counterfactuals,select_positions,complex_iq


def test_systematic_source_selection_has_36_per_RX_without_labels():
    rx=np.repeat(d.RECEIVERS,360);positions=select_positions(rx)
    assert len(positions)==len(set(positions.tolist()))==180
    assert {r:int((rx[positions]==r).sum()) for r in d.RECEIVERS}==dict.fromkeys(d.RECEIVERS,36)
    assert all(np.all(np.diff(positions[i*36:(i+1)*36])>0) for i in range(5))


@pytest.mark.parametrize('rx',[np.zeros(1800),np.repeat(d.RECEIVERS,359),np.array([1]*35+[3]*325+[4]*360+[6]*360+[8]*720)])
def test_wrong_source_role_population_or_missing_RX_capacity_is_rejected(rx):
    with pytest.raises(ValueError):select_positions(rx)


def test_oracle_time_patches_change_only_declared_coordinates_and_do_not_mutate_inputs():
    rng=np.random.default_rng(4);received=rng.normal(size=(3,2,256)).astype(np.float32)
    parent=received+.01;clean=received-.02;before=[x.copy() for x in (received,parent,clean)]
    result=counterfactuals(received,parent,clean)
    assert tuple(result)==d.READOUTS
    assert np.array_equal(result['identity'],received) and np.array_equal(result['parent_R03'],parent)
    assert np.array_equal(result['clean_tail24'][:,:,:232],parent[:,:,:232])
    assert np.array_equal(result['clean_tail24'][:,:,232:],clean[:,:,232:])
    assert np.array_equal(result['clean_prefix232'][:,:,:232],clean[:,:,:232])
    assert np.array_equal(result['clean_prefix232'][:,:,232:],parent[:,:,232:])
    assert np.array_equal(result['clean_reference'],clean)
    assert all(np.array_equal(a,b) for a,b in zip((received,parent,clean),before))


def test_complex_gain_and_frequency_oracles_use_clean_reference_and_preserve_remaining_bins():
    rng=np.random.default_rng(8);clean=rng.normal(size=(2,2,256)).astype(np.float32)
    reference=complex_iq(clean);shifted=(2*np.exp(.4j)*reference)
    parent=np.stack((shifted.real,shifted.imag),1).astype(np.float32)
    result=counterfactuals(parent,parent,clean)
    assert np.allclose(result['clean_gain'],clean,atol=1e-6)
    spectrum=np.fft.fft(reference,norm='ortho');old=np.fft.fft(complex_iq(parent),norm='ortho')
    order=np.argsort(abs(spectrum)**2,axis=-1,kind='stable')
    for name,index in (('clean_low64',order[:,:64]),('clean_high64',order[:,-64:])):
        expected=old.copy();np.put_along_axis(expected,index,np.take_along_axis(spectrum,index,-1),-1)
        actual=np.fft.fft(complex_iq(result[name]),norm='ortho')
        assert np.allclose(actual,expected,atol=1e-6,rtol=1e-6)


@pytest.mark.parametrize('bad',['length','dtype','nan'])
def test_invalid_source_counterfactual_inputs_fail(bad):
    clean=np.ones((2,2,256),np.float32);parent=clean.copy()
    if bad=='length':parent=parent[:,:,:255]
    elif bad=='dtype':parent=parent.astype(np.float64)
    else:parent[0,0,0]=np.nan
    with pytest.raises(ValueError):counterfactuals(clean,parent,clean)


def packets(root):
    saved=root/'prediction';saved.mkdir();ids=np.asarray([str(i) for i in range(d.COUNT)])
    done=dict(status='SOURCE_PREDICTIONS_COMPLETE',role='source_V',views=list(d.VIEWS),count=d.COUNT,
        F_seeds=list(d.F_SEEDS),readouts=list(d.READOUTS),oracle_diagnostic_only=True,formal_target_predictions=False,
        runtime_supported=False,source_selection=False,TX_label_reads=0,target_reads=0,state_updates=0)
    (saved/'complete.json').write_text(json.dumps(done))
    values=dict(ids=ids,seeds=np.asarray(d.F_SEEDS),**{name:np.zeros((d.COUNT,3),np.uint8) for name in d.READOUTS})
    for view in d.VIEWS:
        np.savez(saved/(view+'.npz'),**values)
        for name in d.READOUTS:np.save(saved/(view+'_'+name+'_iq.npy'),np.zeros((d.COUNT,2,256),np.float32))
    return saved,ids,done,values


def test_complete_7view_8readout_3F_source_diagnostic_packet(tmp_path):
    _,ids,_,_=packets(tmp_path)
    assert set(runner.completed_predictions(tmp_path,ids))==set(d.VIEWS)


@pytest.mark.parametrize('bad',['earlylabels','targetreads','formalclaim','runtimeclaim','missingview','missingreadout','stateupdate','badseed','ids','class','missingIQ','nanIQ'])
def test_illegal_or_incomplete_source_predictions_block_truth(tmp_path,bad):
    saved,ids,done,values=packets(tmp_path);view=d.VIEWS[-1]
    if bad=='earlylabels':done['TX_label_reads']=1
    elif bad=='targetreads':done['target_reads']=1
    elif bad=='formalclaim':done['formal_target_predictions']=True
    elif bad=='runtimeclaim':done['runtime_supported']=True
    elif bad=='missingview':done['views']=list(d.VIEWS[:-1])
    elif bad=='missingreadout':done['readouts']=list(d.READOUTS[:-1])
    elif bad=='stateupdate':done['state_updates']=1
    elif bad=='badseed':values['seeds']=np.array([0,1,2])
    elif bad=='ids':values['ids']=ids[::-1]
    elif bad=='class':values['clean_tail24'][0,0]=6
    elif bad=='missingIQ':np.save(saved/(view+'_clean_gain_iq.npy'),np.zeros((1,2,256),np.float32))
    else:np.save(saved/(view+'_clean_gain_iq.npy'),np.full((d.COUNT,2,256),np.nan,np.float32))
    (saved/'complete.json').write_text(json.dumps(done));np.savez(saved/(view+'.npz'),**values)
    with pytest.raises(ValueError):runner.completed_predictions(tmp_path,ids)


def test_independent_scorer_never_opens_source_labels_before_predictions_complete(tmp_path,monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','');saved,ids,done,_=packets(tmp_path)
    done['views']=[];(saved/'complete.json').write_text(json.dumps(done))
    fake=SimpleNamespace(rx=np.ones(d.COUNT),day=np.ones(d.COUNT));positions=np.arange(d.COUNT)
    monkeypatch.setattr(runner,'source_inputs',lambda args:(ids,fake,positions,positions))
    with pytest.raises(ValueError):runner.score(SimpleNamespace(output=tmp_path,source_cache=tmp_path/'NO_LABEL_FILE'))


def test_controller_holds_both_existing_CPU_locks_through_independent_child_score(tmp_path,monkeypatch):
    for run in d.LOCK_RUNS:(tmp_path/run).mkdir()
    locks=[];commands=[]
    monkeypatch.setitem(sys.modules,'fcntl',SimpleNamespace(LOCK_EX=2,flock=lambda handle,mode:locks.append(handle)))
    monkeypatch.setattr(runner,'code_commit',lambda:'x');monkeypatch.setattr(runner,'source_inputs',lambda args:None)
    monkeypatch.setattr(runner,'smoke',lambda args:{});monkeypatch.setattr(runner,'predict',lambda args,models:None)
    def child(command,**kwargs):
        assert len(locks)==2 and all(not handle.closed for handle in locks)
        assert command[2:5]==['-m','experiments.pamf_oracle_attribution.runner','score']
        assert kwargs['env']['CUDA_VISIBLE_DEVICES']=='';commands.append(command)
    monkeypatch.setattr(runner.subprocess,'run',child)
    args=SimpleNamespace(output=tmp_path/d.RUN,expected_commit='x',source_cache=tmp_path/'source',parent_cache=tmp_path/'cache',
        parent_restoration=tmp_path/d.PARENT_RUN,baseline_source_run=tmp_path/d.BASELINE_RUN)
    runner.run(args)
    assert all(handle.closed for handle in locks) and len(commands)==1
    assert json.loads((args.output/'complete.json').read_text())['formal_target_predictions'] is False
