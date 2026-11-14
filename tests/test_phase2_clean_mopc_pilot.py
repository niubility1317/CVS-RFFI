import json
import pytest
from experiments.pamf_phase2_practical.clean_mopc_pilot import verify_diagnostics, reserve_pilot, DIAGNOSTICS, run


def test_diagnostic_gate_rejects_missing_failure_and_query_access(tmp_path):
    source=tmp_path/'source_root';tasks=[];events={}
    for gpu,cap,seed,k,n in DIAGNOSTICS:
        tasks.append(dict(gpu=gpu,capsule=cap,seed=seed,k=k,n=n))
        complete=dict(event='DIAGNOSTIC_COMPLETE',semantics='public_code',updates=20*((5*k+15)//16)*4,capsule=cap,seed=seed,k=k,n=n,
                      source_checkpoint=str(source/'source'/f'mopc_hr-s{seed}'/'source.pt'),mopc_epochs=20,
                      stage_size=5,gradient_clipping=False,query_read=False,truth_read=False,predictions_written=False)
        events[gpu]=complete;(tmp_path/f'gpu{gpu}.log').write_text(json.dumps(complete)+'\n')
    (tmp_path/'owner.json').write_text(json.dumps({'tasks':tasks}))
    assert verify_diagnostics(tmp_path,source)==3
    (tmp_path/'gpu6.log').write_text('Traceback (most recent call last):\n')
    with pytest.raises(ValueError,match='Diagnostic'):
        verify_diagnostics(tmp_path,source)
    (tmp_path/'gpu6.log').write_text(json.dumps(dict(events[6],query_read=True))+'\n')
    with pytest.raises(ValueError,match='Diagnostic'):
        verify_diagnostics(tmp_path,source)
    (tmp_path/'gpu6.log').write_text(json.dumps(dict(events[6],capsule=DIAGNOSTICS[0][1]))+'\n')
    with pytest.raises(ValueError,match='Diagnostic'):
        verify_diagnostics(tmp_path,source)


def test_pilot_identity_has_one_owner_across_different_plans(tmp_path):
    (tmp_path/'preparations').mkdir()
    reserve_pilot(tmp_path,tmp_path/'plan1',tmp_path/'run1')
    with pytest.raises(FileExistsError):
        reserve_pilot(tmp_path,tmp_path/'plan2',tmp_path/'run2')


def test_failed_diagnostics_do_not_reserve_pilot(tmp_path, monkeypatch):
    project=tmp_path/'project';(project/'preparations').mkdir(parents=True)
    diagnostic=tmp_path/'diagnostics';diagnostic.mkdir()
    (diagnostic/'owner.json').write_text(json.dumps({'tasks':[{'gpu':i,'pid':10+i} for i in (5,6,7)]}))
    runtime=tmp_path/'runtime.json';runtime.write_text(json.dumps({'project':str(project)}))
    release=tmp_path/'release';release.mkdir()
    monkeypatch.setattr('experiments.pamf_phase2_practical.clean_mopc_pilot.process_live',lambda _pid:False)
    with pytest.raises(ValueError,match='Diagnostic'):
        run(diagnostic,runtime,tmp_path/'source',tmp_path/'plan',tmp_path/'run',release)
    assert not (project/'preparations/phase2-clean-native-mopc-pilot-owner').exists()
