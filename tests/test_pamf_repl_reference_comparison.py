import unittest
import gzip
import json
import tempfile
from pathlib import Path
from experiments.pamf_repl_reference_comparison import comparison_plan,dominance_stats


class ReferenceComparisonTests(unittest.TestCase):
    def test_complete_dominance_telemetry_and_cap_violation(self):
        def group(ratio):
            return dict(reference_dominance_checked=True,reference_extra_cap=.25,
                reference_extra_ratio=ratio,reference_dominance_cosine=.99,
                reference_zero_fallback=False,reference_rounding_fallback=False)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for end,steps in ((30,2220),(50,4440)):
                folder=root/f'segment{end}';folder.mkdir()
                (folder/'repl_state.json').write_text(json.dumps({'U_parameter_step_guard':
                    {'reference_gradient_EMA_updates':steps,'reference_gradient_beta':.9}}),encoding='utf-8')
                with gzip.open(folder/'step_guard_metrics.jsonl.gz','wt',encoding='utf-8') as stream:
                    for step in range(1,steps+1):
                        stream.write(json.dumps(dict(step=step,groups={g:group(.2) for g in ('t2','cls_head','other')},
                            reference={'current_full_L_vs_reference_gradient':
                                {g:{'cosine':-.2} for g in ('t2','cls_head','other')}}))+'\n')
            facts=dominance_stats(root,.25)
            self.assertEqual(facts['t2']['ratio_defined'],6660)
            self.assertEqual(facts['t2']['reference_rounding_fallback'],0)
            ema=dominance_stats(root,.25,'EMA')
            self.assertEqual(ema['t2']['current_L_vs_EMA_conflict'],6660)
            with gzip.open(root/'segment30'/'step_guard_metrics.jsonl.gz','wt',encoding='utf-8') as stream:
                stream.write(json.dumps(dict(step=1,groups={g:group(.3) for g in ('t2','cls_head','other')}))+'\n')
            with self.assertRaisesRegex(ValueError,'exceeds'):
                dominance_stats(root,.25)

    def test_all_matched_effects_are_predeclared(self):
        rows=comparison_plan()
        self.assertEqual(len(rows),48)
        self.assertEqual({r['comparison'] for r in rows},
            {'CE_vs_full','full_vs_dual','CE_vs_dual','EMA_vs_full','EMA_vs_CE','EMA_vs_dual'})
        self.assertEqual({r['setting'] for r in rows},{'ten_class','source_rx_6'})
        self.assertEqual({(r['weight'],r['cap']) for r in rows},{(w,c) for w in (3.,6.) for c in (.25,.5)})
        self.assertEqual(len({(r['comparison'],r['setting'],r['weight'],r['cap']) for r in rows}),48)


if __name__=='__main__':unittest.main()
