import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def test_pilot_is_one_exact_pending_low_memory_training_row():
    assert importlib.util.find_spec('experiments.pamf_split_spectrum_paper.speed_pilot') is not None
    from experiments.pamf_split_spectrum_paper.speed_pilot import select_pilot
    row = {'id':'train__main_split-branch-ce-no_path_no_spectrum-s2026092702','kind':'train'}
    assert select_pilot([row], gpu=4, own=0, free_mib=9262, busy=80, io=1, mem_kib=200*1024**2, used=False) == row
    for patch in ({'gpu':7}, {'own':1}, {'free_mib':8499}, {'busy':85}, {'io':5}, {'mem_kib':63*1024**2}, {'used':True}):
        args=dict(gpu=4,own=0,free_mib=9262,busy=80,io=1,mem_kib=200*1024**2,used=False)
        args.update(patch)
        assert select_pilot([row], **args) is None
    assert select_pilot([dict(row,id='train__other')],gpu=4,own=0,free_mib=9262,busy=80,io=1,mem_kib=200*1024**2,used=False) is None


def test_process_ownership_requires_exact_release_and_arguments():
    from experiments.pamf_split_spectrum_paper.speed_pilot import matches_dispatcher, matches_worker, command, limits
    rt={'source_release':'/release/source','python':'/env/python'}
    job={'config':'/run/config.json','kind':'train'}
    worker=command(rt,job)
    assert matches_worker(rt,job,'/release/source',worker)
    assert not matches_worker(rt,job,'/release/other',worker)
    assert not matches_worker(rt,job,'/release/source',worker+['unexpected'])
    dispatcher=['/env/python','-u','-m','experiments.pamf_split_spectrum_paper.runner','dispatch','--config','/release/source/runtime.json']
    assert matches_dispatcher(rt,'/release/source',dispatcher)
    assert not matches_dispatcher(rt,'/release/source',dispatcher[:-1]+['/other/runtime.json'])
    assert [limits(i) for i in range(8)]==[3,3,3,3,3,3,3,1]
