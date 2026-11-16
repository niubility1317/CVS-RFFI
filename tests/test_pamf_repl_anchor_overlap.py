import gzip
import json
import tempfile
import unittest
from pathlib import Path
from experiments.pamf_repl_dual_ce.analyze_anchor_overlap import category,analyze


class DualAnchorOverlapTests(unittest.TestCase):
    def test_conflict_categories_and_finite_values(self):
        self.assertEqual(category(-.1,-.2),'both')
        self.assertEqual(category(-.1,.2),'full_only')
        self.assertEqual(category(.1,-.2),'ce_only')
        self.assertEqual(category(0.,0.),'neither')
        self.assertEqual(category(None,.2),'undefined')
        with self.assertRaisesRegex(ValueError,'Finite'):category(float('nan'),0.)

    def test_complete_trajectory_and_zero_step_are_counted(self):
        with tempfile.TemporaryDirectory() as tmp:
            source=Path(tmp)
            common=dict(status='SOURCE_FROZEN',target_access=False,U_truth_read=False)
            (source/'complete.json').write_text(json.dumps(dict(**common,epoch=50,updates=11100,
                config={'row_id':'example'})),encoding='utf-8')
            directions=[(-.2,-.1),(-.2,.1),(.2,-.1),(.2,.1),(None,.1)]
            for end,steps in ((30,2220),(50,4440)):
                folder=source/f'segment{end}';folder.mkdir()
                (folder/'complete.json').write_text(json.dumps(dict(**common,updates=steps)),encoding='utf-8')
                with gzip.open(folder/'step_guard_metrics.jsonl.gz','wt',encoding='utf-8') as stream:
                    for step in range(1,steps+1):
                        a,b=directions[(step-1)%5];zero=(step-1)%5==2
                        d=dict(anchor_cosines_before={'L_full':a,'L_clean':b},
                            anchor_cosines_after={'L_full':None if zero else 0.,'L_clean':None if zero else 0.},
                            actual_parameter_check=True,norm_nonexpansion=True,raw_increment_norm=1.,
                            applied_increment_norm=0. if zero else .5,quantization_fallback=zero,
                            retention=0. if zero else .5)
                        stream.write(json.dumps(dict(step=step,groups={g:d for g in ('t2','cls_head','other')}))+'\n')
            facts=analyze(source)['groups']['t2']
            self.assertEqual(facts['ce_only']['steps'],1332)
            self.assertEqual(facts['ce_only']['zero_extra'],1332)
            self.assertEqual(facts['both']['applied_norm_sum'],666.)
            self.assertEqual(sum(v.get('steps',0) for v in facts.values()),6660)


if __name__=='__main__':unittest.main()
