import pytest
from experiments.pamf_identity_flow.rollout_phase3_profile import summarize,validate_prediction,check_exclusive_gpu
from experiments.pamf_identity_flow import rollout_train as source_train


def test_profile_reports_complete_latency_sample_and_batch_throughput():
    result=summarize([1.,2.,3.,4.],batch=2)
    assert result['iterations']==4
    assert result['median_ms']==pytest.approx(2.5)
    assert result['throughput_samples_s']==pytest.approx(800.)
    assert result['p95_ms']>=result['median_ms']
    with pytest.raises(ValueError):summarize([],batch=2)
    with pytest.raises(ValueError):summarize([1.,float('nan')],batch=2)


def test_profile_binds_runtime_and_checkpoint_to_completed_predictions(tmp_path):
    run=tmp_path/'run';runtime=tmp_path/'runtime.json';checkpoint=run/source_train.ARM/'final.pt'
    value=dict(status='IDENTITY_ROLLOUT_PHASE3_PREDICTIONS_COMPLETE',run_id=source_train.RUN,
        candidate_code_commit='a'*40,private_truth_reads=0,
        counts=dict(known=27000,novel=3150,clean=10238,star=10238),
        config=dict(reference_runtime=str(runtime),candidate_checkpoint=str(checkpoint)))
    validate_prediction(value,run,runtime,'a'*40)
    with pytest.raises(ValueError):validate_prediction(value,run,tmp_path/'other.json','a'*40)
    with pytest.raises(ValueError):validate_prediction(value,run,runtime,'b'*40)


def test_profile_rejects_other_gpu_compute_process(monkeypatch):
    from experiments.pamf_identity_flow import rollout_phase3_profile as profile
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','4')
    monkeypatch.setattr(profile.os,'getpid',lambda:123)
    def output(command,**kwargs):
        return '4, GPU-good\n' if '--query-gpu=index,uuid' in command else 'GPU-good, 123\nGPU-other, 99\n'
    monkeypatch.setattr(profile.subprocess,'check_output',output)
    assert check_exclusive_gpu()==[123]
    def contaminated(command,**kwargs):
        return '4, GPU-good\n' if '--query-gpu=index,uuid' in command else 'GPU-good, 123\nGPU-good, 456\n'
    monkeypatch.setattr(profile.subprocess,'check_output',contaminated)
    with pytest.raises(RuntimeError,match='contaminated'):check_exclusive_gpu()
