import unittest

from experiments.pamf_phase2_practical.gpu_limit_state import partition, stable_workers


class TakeoverStateTest(unittest.TestCase):
    def test_unidentified_transient_child_is_reconciled_or_fails_closed(self):
        observed = iter([{}, {'worker': {'pid': 5}}])
        def check(value):
            if not value: raise ValueError('Unidentified live controller child: 5')
        self.assertIn('worker', stable_workers(lambda: next(observed), check, lambda _: None))
        with self.assertRaisesRegex(ValueError, 'Unidentified live controller child'):
            stable_workers(lambda: {}, check, lambda _: None, attempts=2)
        with self.assertRaisesRegex(ValueError, 'Other failure'):
            stable_workers(lambda: {}, lambda _: (_ for _ in ()).throw(ValueError('Other failure')), lambda _: None)

    def test_failed_profile_and_dependencies_are_preserved(self):
        def row(profile, deps=()):
            return {'origin': 'legacy', 'profile': profile, 'dependencies': list(deps)}
        by = {'failed': row('issl'), 'blocked_profile': row('issl'),
              'blocked_dependency': row('other', ['failed']),
              'blocked_transitively': row('third', ['blocked_dependency']),
              'finished': row('other'), 'active': row('other'), 'ready': row('other')}
        remaining, blocked, errors = partition(by, {'active'}, {'finished'}, {'failed'})
        self.assertEqual(remaining, {'ready'})
        self.assertEqual(blocked, {'blocked_profile', 'blocked_dependency', 'blocked_transitively'})
        self.assertEqual(errors, {('legacy', 'issl')})

    def test_new_worker_gpu_set_is_exact(self):
        import ast
        from pathlib import Path
        source = Path('experiments/pamf_phase2_practical/gpu_limit_takeover.py').read_text(encoding='utf-8')
        tree = ast.parse(source)
        gpu_loops = [n for n in ast.walk(tree) if isinstance(n, ast.For)
                     and isinstance(n.iter, ast.Subscript) and isinstance(n.iter.value, ast.Name)
                     and n.iter.value.id == 'rt' and isinstance(n.iter.slice, ast.Constant)
                     and n.iter.slice.value == 'allowed_gpus']
        self.assertEqual(len(gpu_loops), 1)


if __name__ == '__main__': unittest.main()
