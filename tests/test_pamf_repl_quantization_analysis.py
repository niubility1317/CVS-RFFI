import gzip
import json
import tempfile
import unittest
from pathlib import Path
from experiments.pamf_repl_dual_ce.analyze_quantization import aggregate


class QuantizationAnalysisTests(unittest.TestCase):
    def test_fallback_magnitude_uses_raw_not_applied(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)
            (path/'complete.json').write_text(json.dumps(dict(status='SOURCE_FROZEN',updates=222,
                target_access=False,U_truth_read=False)),encoding='utf-8')
            (path/'metrics_epoch.jsonl').write_text(json.dumps(dict(lr=1e-6))+'\n',encoding='utf-8')
            with gzip.open(path/'step_guard_metrics.jsonl.gz','wt',encoding='utf-8') as stream:
                for i in range(1,223):
                    fallback=i==1
                    group=dict(raw_increment_norm=2. if fallback else 1.,applied_increment_norm=0. if fallback else .5,
                               quantization_fallback=fallback,actual_parameter_check=True,norm_nonexpansion=True)
                    stream.write(json.dumps(dict(step=i,groups={g:group for g in ('t2','cls_head','other')}))+'\n')
            result=aggregate(path,offset=49,steps=222)['50']['groups']['t2']
            self.assertEqual(result['fallback_steps'],1)
            self.assertEqual(result['fallback_raw_norm_sum'],2.)
            self.assertEqual(result['raw_norm_sum'],223.)
            self.assertEqual(result['applied_norm_sum'],110.5)
            (path/'complete.json').write_text(json.dumps(dict(status='SOURCE_FROZEN',updates=222,
                target_access=True,U_truth_read=False)),encoding='utf-8')
            with self.assertRaisesRegex(ValueError,'source-only'):aggregate(path,offset=49,steps=222)


if __name__=='__main__':unittest.main()
