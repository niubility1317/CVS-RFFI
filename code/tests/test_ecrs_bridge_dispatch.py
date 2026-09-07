import importlib.util
from pathlib import Path
import sys

SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location('bridge_dispatch', SCRIPTS / 'run_ecrs_v2_bridge_batch_20260908.py')
dispatch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dispatch)


def test_context_supervisor_does_not_consume_training_slot():
    assert dispatch.occupied_slots({0: 'a', 1: 'b'}, [('a', 10), ('a', 11), ('b', 12)],
                                   ignored_pids={10}) == {0: 1, 1: 1}


def test_new_child_reserves_capacity_before_cuda_is_visible():
    counts = dispatch.occupied_slots({0: 'a'}, [('a', 11)], reservations=[(0, 12)])
    assert counts[0] == 2


def test_visible_child_is_not_double_counted_and_unknown_compute_is_retained():
    counts = dispatch.occupied_slots({0: 'a', 1: 'b'}, [('a', 11), ('a', 11), ('b', 99)],
                                    reservations=[(0, 11)])
    assert counts == {0: 1, 1: 1}


def test_queue_is_exactly_four_predeclared_rows_without_fusion():
    assert dispatch.ROWS == ['B3a', 'B3b', 'B3c', 'B4']
