from copy import deepcopy
from pathlib import Path
import signal
import tempfile
import unittest
from unittest.mock import patch

from experiments.cric_dura_comparison.io import write, read
from experiments.pamf_ssl_research.queued_replacement import validate_queued, replace
from experiments.pamf_ssl_research.view_anchor import RUN, PREVIOUS_RUN, RESOURCE_GPU_BY_SEED
from experiments.pamf_ssl_research.plan import SEEDS
from experiments.pamf_ssl_research.control import validate_rows


def fixture(root):
    old = dict(run_id=PREVIOUS_RUN, launch_owner='codex-pamf-view-anchor-20261023',
        run_root=str(root/'old_run'), release_root=str(root/'old_release'),runtime_file=str(root/'old_runtime.json'),
        rows=[dict(row_id='VIEWANCHOR-s'+str(seed)) for seed in SEEDS])
    new = dict(old, run_id=RUN, replaces_queued_runtime=old['runtime_file'])
    state = dict(status='WAITING_FOR_SMOKE_GPU',controller_pid=91,
                 jobs={row['row_id']:dict(status='QUEUED') for row in old['rows']})
    proc = dict(pid=91, alive=True,state='S',cwd=old['release_root'],children=[],
        argv=['python','-m','experiments.pamf_ssl_research.control','--config',old['runtime_file']])
    return old,new,state,proc


class QueuedResourceReplacementTest(unittest.TestCase):
    def test_controller_smoke_uses_relocated_physical_card_five(self):
        from experiments.pamf_ssl_research.control import run
        class SmokeReached(Exception):
            pass
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            rt=dict(methods=['VIEWANCHOR'],gpu_by_seed={str(seed):gpu for seed,gpu in RESOURCE_GPU_BY_SEED.items()},
                rows=[dict(row_id='VIEWANCHOR-s'+str(seed),method='VIEWANCHOR',model_seed=seed,
                           gpu=RESOURCE_GPU_BY_SEED[seed]) for seed in SEEDS],
                run_root=str(root/'run'),release_root=str(root),runtime_file=str(root/'runtime.json'),code_commit='c')
            def smoke(command,**kwargs):
                self.assertEqual(kwargs['env']['CUDA_VISIBLE_DEVICES'],'5')
                self.assertIn('experiments.pamf_ssl_research.view_anchor_worker',command)
                raise SmokeReached
            with patch('experiments.pamf_ssl_research.control.capacity',return_value=(1,23000)) as capacity, \
                 patch('experiments.pamf_ssl_research.control.subprocess.run',side_effect=smoke):
                with self.assertRaises(SmokeReached):
                    run(rt)
                capacity.assert_called_once_with(5)
            state=read(root/'run/state.json')
            self.assertEqual(state['gpu_by_seed'],{str(seed):gpu for seed,gpu in RESOURCE_GPU_BY_SEED.items()})

    def test_active_child_wrong_identity_or_smoke_is_never_stopped(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            old,new,state,proc=fixture(root)
            validate_queued(old,new,state,proc)
            for key,value in (('children',['999']),('cwd','another-release'),('argv',['foreign-process'])):
                with self.assertRaises(ValueError):
                    validate_queued(old,new,state,dict(proc,**{key:value}))
            bad=deepcopy(state)
            bad['jobs'][old['rows'][0]['row_id']]=dict(status='RUNNING',pid=99)
            with self.assertRaises(ValueError):
                validate_queued(old,new,bad,proc)
            Path(old['run_root']).mkdir()
            write(Path(old['run_root'])/'smoke.json',dict(status='PASS'))
            with self.assertRaises(ValueError):
                validate_queued(old,new,state,proc)

    def test_verified_unstarted_controller_only_then_preserved_terminal_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            old,new,state,proc=fixture(root)
            write(Path(old['runtime_file']),old)
            write(Path(old['run_root'])/'state.json',state)
            write(Path(old['release_root'])/'submit.json',dict(pid=91))
            with patch('experiments.pamf_ssl_research.queued_replacement.process',
                       side_effect=[proc,dict(proc,state='T'),dict(pid=91,alive=False)]), \
                 patch('experiments.pamf_ssl_research.queued_replacement.os.kill') as kill, \
                 patch.object(signal,'SIGSTOP',19,create=True),patch.object(signal,'SIGCONT',18,create=True):
                result=replace(new)
            self.assertEqual([call.args[0] for call in kill.call_args_list],[91,91,91])
            self.assertEqual([call.args[1] for call in kill.call_args_list],[19,signal.SIGTERM,18])
            self.assertEqual(result['old_GPU_workers'],0)
            saved=read(Path(old['run_root'])/'state.json')
            self.assertEqual(saved['status'],'REPLACED_BEFORE_GPU_RESOURCE_CHANGE')
            self.assertTrue(Path(old['run_root']).is_dir())

    def test_relocated_seed_cards_are_explicit_and_original_map_is_rejected(self):
        rows=[dict(row_id='VIEWANCHOR-s'+str(seed),method='VIEWANCHOR',model_seed=seed,
                   gpu=RESOURCE_GPU_BY_SEED[seed]) for seed in SEEDS]
        validate_rows(rows,('VIEWANCHOR',),RESOURCE_GPU_BY_SEED)
        with self.assertRaises(ValueError):
            validate_rows(rows,('VIEWANCHOR',))
        with self.assertRaises(ValueError):
            validate_rows(rows,('VIEWANCHOR',),{seed:5 for seed in SEEDS})
