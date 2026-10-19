"""Reproduce an occupied card between dispatch and worker startup, without CUDA."""
import ast
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace, ModuleType
from unittest.mock import patch
import sys
import unittest


class CapacityRace(unittest.TestCase):
    def test_contended_worker_waits_before_training(self):
        source = Path(__file__).with_name('dispatch.py').read_text(encoding='utf8')
        worker = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == 'worker')
        observations = [dict(pids={100, 200}, free_mb=20000), dict(pids={100}, free_mb=20000)]
        calls, sleeps, records = [], [], []
        @contextmanager
        def lock(*args, **kw):
            yield
        def capacity():
            state = observations.pop(0)
            calls.append(state)
            return {4: state}
        d = SimpleNamespace(rows=lambda: [dict(row_id='CE_HALF-s2026092702')], config=lambda r: r,
                            GPU_IDS=(4, 6), BASE=Path('/run'), PROJECT='/project',
                            write=lambda p, x: records.append(x))
        observer = ModuleType('ce_half_capacity_fixture.observer')
        observer.train = lambda c: calls.append('TRAIN')
        package = ModuleType('ce_half_capacity_fixture')
        package.__path__ = []
        env = dict(__package__='ce_half_capacity_fixture', d=d, lock=lock, Path=Path,
                   os=SimpleNamespace(environ={'CUDA_VISIBLE_DEVICES': '4'}, getpid=lambda: 100),
                   capacity=capacity, time=SimpleNamespace(sleep=lambda s: sleeps.append(s), time=lambda: 0),
                   execute=lambda *a: calls.append('PREDICT'))
        exec(compile(ast.Module(body=[worker], type_ignores=[]), '<actual worker>', 'exec'), env)
        with patch.dict(sys.modules, {'ce_half_capacity_fixture': package, 'ce_half_capacity_fixture.observer': observer}):
            env['worker']('CE_HALF-s2026092702')
        self.assertEqual(len(observations), 0)
        self.assertEqual(calls[-2:], ['TRAIN', 'PREDICT'])
        self.assertEqual(len(sleeps), 1)
        self.assertEqual([x['status'] for x in records], ['WAITING_CAPACITY', 'CAPACITY_READY'])


if __name__ == '__main__':
    unittest.main()
