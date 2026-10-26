import unittest
from experiments.pamf_channel_pl.rgpl_source_graph import RUN,RELEASE,validate_runtime,teacher_folder,NAMESPACES


class SourceGraphBoundaryTests(unittest.TestCase):
    def test_runtime_has_only_fixed_source_diagnostic_handles(self):
        c=dict(run_id=RUN,project='/project',python='python',release_root='/project/releases/'+RELEASE,run_root='/project/runs/'+RUN,
            context='context',runtime_file='runtime',code_commit='commit',frozen_sources={},allowed_gpus=list(range(8)),launch_owner='owner')
        validate_runtime(c)
        for extra in (dict(target_inputs='query'),dict(truth='labels'),dict(run_root='/foreign'),dict(allowed_gpus=[0]),dict(run_id='other')):
            with self.subTest(extra=extra),self.assertRaises(ValueError):validate_runtime(dict(c,**extra))
        self.assertEqual(teacher_folder('/project'),'/project/runs/20261026-phase1-PAMF-NoSIG-RGPL-base-source-m2-r01/F/F0-NoSIG-PL-s2026092701/source/segment30')
        self.assertEqual(NAMESPACES,('PAMF/restored_dose/U/fixed',)+tuple('PAMF/realization_diversity/U/fresh'+str(r) for r in (1,2,3)))


if __name__=='__main__':unittest.main()
