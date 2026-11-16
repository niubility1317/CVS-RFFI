import gzip,json,tempfile,unittest
from pathlib import Path
from experiments.pamf_repl_step_guard.analyze_training import analyze,FIELDS,GROUPS


def fixture(root,broken=False):
    root=Path(root)
    (root/'complete.json').write_text(json.dumps(dict(status='SOURCE_FROZEN',epoch=50,updates=11100,
        target_access=False,U_truth_read=False,config={'row_id':'test'})))
    for end,offset,steps in ((30,20,2220),(50,30,4440)):
        folder=root/f'segment{end}';folder.mkdir()
        (folder/'repl_state.json').write_text(json.dumps(dict(calls=steps,seen=steps,accepted=steps,
            U_truth_read=False,U_parameter_step_guard={'calls':steps})))
        (folder/'metrics_epoch.jsonl').write_text(''.join(json.dumps(dict(epoch=i,val_tx_acc=90.))+'\n'
                                                        for i in range(1,end-offset+1)))
        paths=[folder/'step_metrics.jsonl.gz',folder/'step_guard_metrics.jsonl.gz',folder/'U_pseudo_observations.jsonl.gz']
        streams=[gzip.open(path,'wt') for path in paths]
        try:
            for i in range(1,steps+1):
                metrics={key:0. for key in FIELDS};metrics.update(HPL_seen=i,HPL_accepted=i)
                group=dict(actual_parameter_check=True,norm_nonexpansion=True,
                    anchor_cosines_before={'L_full':-.1},anchor_cosines_after={'L_full':0.},
                    retention=.8,quantization_fallback=False)
                rows=[dict(step=i,metrics=metrics),dict(step=i,groups={g:group for g in GROUPS}),
                      dict(step=offset*222+i+(1 if broken and i==1 else 0),physical_ids=['x'],
                           pseudo=[0],confidence=[.99],accepted=[True])]
                for stream,row in zip(streams,rows):stream.write(json.dumps(row)+'\n')
        finally:
            for stream in streams:stream.close()


class FullAnalysisTests(unittest.TestCase):
    def test_all_6660_steps_and_unique_coverage(self):
        with tempfile.TemporaryDirectory() as root:
            fixture(root);result=analyze(root)
            self.assertEqual((result['steps'],result['U_seen'],result['U_accepted']),(6660,6660,6660))
            self.assertEqual(result['unique_U_accepted'],1)
            self.assertEqual(len(result['epochs']),30)
            self.assertEqual(result['groups']['t2']['conflict_before'],6660)

    def test_rejects_unpaired_steps(self):
        with tempfile.TemporaryDirectory() as root:
            fixture(root,broken=True)
            with self.assertRaises(ValueError):analyze(root)


if __name__=='__main__':unittest.main()
