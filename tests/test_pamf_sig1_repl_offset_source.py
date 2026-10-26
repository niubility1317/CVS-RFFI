from copy import deepcopy
from unittest.mock import patch
import unittest
from experiments.pamf_channel_pl import sig1_repl_offset_source as m
from experiments.pamf_channel_pl import sig1_repl_base as base


def context():
    return dict(project='/project',python='python',frozen_sources={},G=dict(kind='original_F0_LU',roles='LU'),
        cases=[dict(row_id='F0-ordinary_PL-s'+str(s),protocol='F0',model_seed=s,parent_source='/old/segment20',
            source_config=dict(ablation='F21',num_classes=6),expected_source_contract='roles') for s in (2026092701,2026092702,2026092703)])


def cfg(method):
    old=dict(num_classes=6,source_rxs=[1,3,4,6,8],source_days=[1,2,3],source_cache='cache',source_contract='roles',
        hard_pl=dict(method='HPLT1U',threshold=.95,weight=6.),hard_pl_no_uas=True,hard_pl_label_smoothing=.01,hard_pl_shared_style_plan=True,
        hard_pl_mixstyle=dict(probability_per_level=.5,levels=['time_down','t2.pool'],alpha=.1,eps=1e-6),
        hard_pl_gradient_guard=dict(U_to_L_norm_cap=1.),hard_pl_joint_contrast=dict(weight=.05,CE_normalizer=1.25),
        hard_pl_receiver_extrapolation=dict(alpha=.25,extrapolated_CE_coefficient=.25,CE_normalizer=1.25))
    with patch.object(base,'read',return_value=dict(config=old)),patch.object(m,'read',return_value=dict(config=old)):
        return m.config(context(),'commit',method)


class NewREPLSourceTests(unittest.TestCase):
    def test_exact_source_boundaries_both_references_and_cost_tie(self):
        scores={name:dict(raw_correct=26000,H3_correct=60000,G3_correct=72000) for name in m.REFERENCES+m.METHODS}
        scores[m.REFERENCES[1]]=dict(raw_correct=26010,H3_correct=59990,G3_correct=72010)
        for name in m.METHODS:scores[name]=dict(raw_correct=25983,H3_correct=59919,G3_correct=72253)
        rows,chosen=m.choose_source(scores)
        self.assertTrue(all(r['source_pass'] for r in rows));self.assertEqual(chosen,'REPL_NoStyleNoRXE')
        scores['REPL_RXOffset']['G3_correct']-=1;scores['REPL_NoStyle']['raw_correct']-=1
        rows,chosen=m.choose_source(scores)
        self.assertEqual([r['method'] for r in rows if r['source_pass']],['REPL_NoStyleNoRXE'])
        self.assertEqual(chosen,'REPL_NoStyleNoRXE')

    def test_runtime_rejects_target_and_foreign_outputs(self):
        release='/project/releases/'+m.RELEASE
        rt=dict(run_id=m.RUN,project='/project',python='python',release_root=release,run_root='/project/runs/'+m.RUN,
            context=release+'/context.json',runtime_file=release+'/runtime.json',code_commit='a'*40,frozen_sources={},allowed_gpus=list(range(8)),
            launch_owner='codex-pamf-sig1-repl-offset-source-20261026')
        m.validate_runtime(rt)
        for bad in (dict(rt,truth='query'),dict(rt,run_root='/foreign'),dict(rt,context='/target')):
            with self.assertRaises(ValueError):m.validate_runtime(bad)

    def test_only_three_new_candidates_no_control_training_and_fixed_budget(self):
        for method in m.METHODS:
            c=cfg(method);m.validate_config(c)
            self.assertEqual(c['sig_time_views'],1);self.assertEqual(c['epochs'],50);self.assertEqual(c['optimizer_updates'],11100)
            self.assertFalse(c['repl_row']['blocks']['style']);self.assertTrue(c['repl_row']['blocks']['guard']);self.assertTrue(c['repl_row']['blocks']['joint'])
        for method in ('SIG1-PL','SIG1-Sup','REPL','Reference_PL_X3'):
            with self.assertRaises(ValueError):m.config(context(),'commit',method)

    def test_source_contract_default_head_and_recipe_negative_cases(self):
        c=cfg('REPL_RXOffset')
        for change in (dict(truth='query'),dict(source_context='/target'),dict(sig_time_views=3),dict(U_use=False),dict(default_head_joint=False),
                       dict(output_root='/foreign'),dict(epochs=70),dict(own_EMA=True),dict(receiver_extrapolation_mode='full_pair'),dict(rx_initial_features='/foreign')):
            with self.subTest(change=change),self.assertRaises(ValueError):m.validate_config(dict(c,**change))
        bad=deepcopy(c);bad['hard_pl']['weight']=2.
        with self.assertRaises(ValueError):m.validate_config(bad)
        bad=deepcopy(c);bad['hard_pl_mixstyle']['probability_per_level']=.5
        with self.assertRaises(ValueError):m.validate_config(bad)

    def test_only_own_E20_and_own_segment30_can_be_parents(self):
        c=cfg('REPL_RXOffset')
        s=dict(c,output_root=c['output_root']+'/segment30',segment_epochs=10,epoch_offset=20,step_offset=4440,parent_source=c['shared_SIG1_E20'])
        m.validate_config(s)
        for change in (dict(parent_source='/stopped/control/partial'),dict(step_offset=6660),dict(segment_epochs=20)):
            with self.assertRaises(ValueError):m.validate_config(dict(s,**change))
        s=dict(c,output_root=c['output_root']+'/segment50',segment_epochs=20,epoch_offset=30,step_offset=6660,parent_source=c['output_root']+'/segment30')
        m.validate_config(s)
        with self.assertRaises(ValueError):m.validate_config(dict(s,parent_source=c['shared_SIG1_E20']))


if __name__=='__main__':unittest.main()
