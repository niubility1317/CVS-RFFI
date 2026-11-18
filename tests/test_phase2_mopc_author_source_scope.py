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


def test_native_smoke_rejects_old_full_source_prototypes():
    from experiments.pamf_phase2_practical.mopc_author_satellite_restart import require_author_source_prototypes
    with pytest.raises(ValueError,match='source prototypes'):
        require_author_source_prototypes({'base_metadata':{'source_role':'L_s'}},1)
    require_author_source_prototypes({'base_metadata':{'source_role':'L_s',
        'source_sampling':'author_shuffled_drop_last','source_batch_size':16,'source_rows_used':16,
        'source_rows':25,'source_sampling_seed':1}},1)


def test_native_runtime_uses_new_source_without_mutating_previous():
    from experiments.pamf_phase2_practical.mopc_author_satellite_restart import native_runtime_for_derived_source
    old={'run_root':'/old','source_contract':'/contract','source_cache':'/cache','code_commit':'old'}
    current=native_runtime_for_derived_source(old,'/new/source','/new/release','newcommit')
    assert old['run_root']=='/old' and old['code_commit']=='old'
    assert current['run_root']=='/new/source' and current['release_root']=='/new/release'
    assert current['source_contract']=='/contract' and current['source_cache']=='/cache'
    assert current['code_commit']=='newcommit'
