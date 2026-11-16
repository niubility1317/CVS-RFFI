import gzip,json,tempfile,unittest
from pathlib import Path
from experiments.pamf_repl_step_guard.analyze_batch_association import analyze,available_records


class BatchAssociationTests(unittest.TestCase):
    def test_epoch_quintiles_and_truth_blind_pseudo_distributions(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            paths=[root/k for k in ('step_metrics.jsonl.gz','step_guard_metrics.jsonl.gz','U_pseudo_observations.jsonl.gz')]
            streams=[gzip.open(p,'wt') for p in paths]
            try:
                for i in range(1,223):
                    high=i>177
                    rows=[{'step':i,'metrics':{'train/loss_tx_labeled':i}},
                          {'step':i,'groups':{'t2':{'anchor_cosines_before':{'L_full':-.1 if high else .1}}}},
                          {'step':4440+i,'pseudo':[1 if high else 0],'accepted':[True]}]
                    for stream,row in zip(streams,rows):stream.write(json.dumps(row)+'\n')
            finally:
                for stream in streams:stream.close()
            data=paths[0].read_bytes();paths[0].write_bytes(data[:-4])
            self.assertEqual(len(available_records(paths[0])),222)
            result=analyze(root,step_offset=4440)
            self.assertEqual((result['aligned_steps'],result['full_epochs']),(222,1))
            self.assertEqual(result['CE_quintiles'][0]['conflict_rate'],0.)
            self.assertEqual(result['CE_quintiles'][4]['conflict_rate'],1.)
            self.assertAlmostEqual(result['pseudo_class_js_bits'],1.)
            self.assertFalse(result['U_truth_read'])
            with self.assertRaises(ValueError):analyze(root,step_offset=0)


if __name__=='__main__':unittest.main()
