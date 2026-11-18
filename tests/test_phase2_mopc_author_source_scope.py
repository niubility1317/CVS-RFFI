from experiments.pamf_phase2_practical import clean_source_smoke
import pytest


def test_target_worker_skips_source_iq_when_dispatcher_performs_source_check(monkeypatch):
    seen=[]
    monkeypatch.setattr(clean_source_smoke,'check',lambda rt,bundle:seen.append('source_read'))
    clean_source_smoke.check_worker({}, {}, {'source_smoke_in_dispatch':True})
    assert seen==[]
    clean_source_smoke.check_worker({}, {}, {})
    assert seen==['source_read']


def test_author_restart_documents_source_replay():
    from experiments.pamf_phase2_practical.mopc_author_satellite_restart import SOURCE_CACHE_USE
    assert SOURCE_CACHE_USE['mrior_sda']=='L_s source replay during B'
    assert SOURCE_CACHE_USE['twostage']=='L_s source replay during B'
    assert SOURCE_CACHE_USE['mopc_hr']=='source-only smoke; frozen source prototypes'


def test_dispatch_cannot_skip_worker_source_checks_without_launcher_smoke():
    from experiments.pamf_phase2_practical.satellite_native_c import require_source_smoke_mode
    with pytest.raises(ValueError,match='source'):
        require_source_smoke_mode({'jobs':[{'source_smoke_in_dispatch':True}]})
    require_source_smoke_mode({'author_source_smoke':True,'jobs':[{'source_smoke_in_dispatch':True}]})
