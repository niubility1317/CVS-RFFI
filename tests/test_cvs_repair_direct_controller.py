from experiments.cvs_phase1_repair.direct_controller import idle_only,visible_reservations


def test_never_take_last_slot_or_ignore_pre_cuda_module(tmp_path):
    p=tmp_path/'101';p.mkdir()
    (p/'stat').write_text('101 (python) S 0 0 0')
    (p/'cmdline').write_bytes(b'python\0-m\0experiments.cvs_phase1_stack.source\0')
    (p/'environ').write_bytes(b'CUDA_VISIBLE_DEVICES=0\0')
    result=idle_only({0:dict(pids=set(),free_mb=24000),1:dict(pids={99},free_mb=20000),2:dict(pids=set(),free_mb=24000)},visible_reservations(tmp_path))
    assert result[0]['free_mb']==0 and result[0]['pids']=={101}
    assert result[1]['free_mb']==0 and result[2]['free_mb']==24000


def test_dead_reservation_is_not_a_blocker(tmp_path):
    p=tmp_path/'101';p.mkdir()
    (p/'stat').write_text('101 (python) Z 0 0 0')
    (p/'cmdline').write_bytes(b'python\0')
    (p/'environ').write_bytes(b'CUDA_VISIBLE_DEVICES=0\0')
    assert visible_reservations(tmp_path)=={}


def test_pre_cuda_worker_waits_on_raced_third_reservation(monkeypatch):
    import sys
    from experiments.cvs_phase1_repair import direct_controller
    monkeypatch.setitem(sys.modules,'direct_controller',direct_controller)
    from experiments.cvs_phase1_repair.train_direct import capacity_ready
    caps={0:dict(pids={10,11},free_mb=16000)}
    assert not capacity_ready(caps,{0:{10,11,12}},0,12)[0]
    assert capacity_ready({0:dict(pids={10},free_mb=20000)},{0:{10,12}},0,12)[0]
