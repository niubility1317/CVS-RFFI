import unittest
from experiments.pamf_channel_pl.noisy_hard_source import RUN,RELEASE,validate_runtime,validate_config,raw_labeler
from experiments.pamf_channel_pl.noisy_hard_objective import METHODS,RECIPES
from experiments.pamf_channel_pl.nosig_rgpl_base import prefix
from experiments.pamf_channel_pl.rgpl_source_graph import teacher_folder


class NoisyHardSourceTests(unittest.TestCase):
    def test_fixed_source_runtime_rejects_truth_and_foreign_roots(self):
        c=dict(run_id=RUN,project='/project',python='python',release_root='/project/releases/'+RELEASE,run_root='/project/runs/'+RUN,
            context='context',runtime_file='runtime',code_commit='commit',frozen_sources={},allowed_gpus=list(range(8)),launch_owner='owner')
        validate_runtime(c)
        for extra in (dict(truth='labels'),dict(target_inputs='query'),dict(run_root='/foreign'),dict(allowed_gpus=[0])):
            with self.subTest(extra=extra),self.assertRaises(ValueError):validate_runtime(dict(c,**extra))

    def fixture(self,method):
        project='/project';rid='F0-'+method+'-s2026092701'
        return dict(run_id=RUN,project=project,row_id=rid,paper_method=method,model_seed=2026092701,protocol_id='F0',algorithm_id='pamf_noisy_hard_pl_v1',
            method_id='pamf_t1_e50',num_classes=6,source_rxs=[1,3,4,6,8],source_days=[1,2,3],no_sigtime=True,regime='matched_sat_plain_pl',ablation='F21',
            U_use=True,U_start_epoch=14,plain_PL_threshold=.95,plain_PL_weight=1.,labeled_batch=128,unlabeled_batch=256,
            raw_labeler=raw_labeler(project),noisy_U_cache=project+'/runs/'+RUN+'/noisy_U_cache',noisy_recipe=RECIPES[method],shared_NoSIG_E20=prefix(project),
            epochs=50,optimizer_updates=11100,segment_epochs=20,epoch_offset=30,step_offset=6660,parent_source=teacher_folder(project),
            output_root=project+'/runs/'+RUN+'/F/'+rid+'/source',default_head_joint=True,
            repl_row=dict(structure='NO_SIGNATURE',U=True,L_A=True,L_B=True,blocks=dict(style=False,joint=False,guard=False,extrapolation=False)))

    def test_each_fixed_recipe_and_both_source_output_levels(self):
        self.assertEqual(len(METHODS),6)
        for method in METHODS:
            c=self.fixture(method);validate_config(c);validate_config(dict(c,output_root=c['output_root']+'/segment50'))

    def test_wrong_parent_budget_architecture_or_auxiliary_objective_rejected(self):
        c=self.fixture('NoisyGCEG')
        for change in (dict(parent_source='/foreign'),dict(segment_epochs=10),dict(no_sigtime=False),dict(noisy_recipe=dict(q=.3,rho=.25,extra='G',uniform=False)),
                dict(own_EMA=True),dict(posthoc_head_fit=True),dict(target_inputs='query'),dict(hard_pl_teacher='foreign')):
            with self.subTest(change=change),self.assertRaises(ValueError):validate_config(dict(c,**change))


if __name__=='__main__':unittest.main()
