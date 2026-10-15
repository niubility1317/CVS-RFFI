"""Negative cases for complete-log evidence; no training or model imports."""
import gzip,json,tempfile,unittest
from pathlib import Path
from experiments.cvs_urban_semantic50._audit_worker import parse_steps,audit


class Checks(unittest.TestCase):
    def records(self):
        return [dict(step=i,epoch=(i-1)//2+1,metrics={'urban/clean_CE':float(i),'train/optimizer_step_applied':1,
            'train/grad_clip_active':1,'train/grad_before_clip':6.,'train/grad_total':5.,'train/grad_clip_limit':5.}) for i in range(1,5)]

    def parse(self,rows):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'steps.jsonl.gz'
            with gzip.open(path,'wt',encoding='utf-8') as stream:
                for row in rows:stream.write(json.dumps(row)+'\n')
            return parse_steps(path,epochs=2,steps_per_epoch=2)

    def test_all_steps_are_averaged_and_clip_counts_are_actual(self):
        result=self.parse(self.records());self.assertEqual(result['complete_steps'],4)
        self.assertEqual(result['curves']['urban/clean_CE'],[1.5,3.5]);self.assertEqual(result['clipping']['clipped_steps'],4)

    def test_partial_wrong_sequence_nonfinite_skips_or_hidden_truth_are_rejected(self):
        with self.assertRaises(ValueError):self.parse(self.records()[:-1])
        for key,value in (('train/optimizer_step_applied',0),('train/skipped_nonfinite_grad',1),('train/pseudo_truth_available',1),('urban/clean_CE',float('nan'))):
            rows=self.records();rows[1]['metrics'][key]=value
            with self.assertRaises(ValueError):self.parse(rows)
        rows=self.records();rows[1]['step']=9
        with self.assertRaises(ValueError):self.parse(rows)

    def test_incomplete_matrix_does_not_read_rows_or_target_scores(self):
        with tempfile.TemporaryDirectory() as temp:
            result=audit(dict(base=temp,rows=[dict(source='must-not-be-opened')]))
        self.assertEqual(result['status'],'WAITING_FOR_MATRIX');self.assertEqual(result['rows'],[])
        self.assertFalse(result['reads_target_scores'])


if __name__=='__main__':unittest.main()
