"""Target isolation and immutable outputs; no real target IQ or truth is opened."""
import copy,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from . import design as d,evaluate,report,model


class Checks(unittest.TestCase):
    def records(self):
        def metric(clean,low,urban,worst):return dict(epoch=50,source_only=True,query_read=False,source_count=720,
            clean_accuracy=clean,scene_accuracy={'practical_low_urban':low},urban_mean=urban,worst_urban_RX=worst)
        return dict(CLEAN=metric(.98,.4,.5,.4),NOMINAL=metric(.97,.66,.77,.7),POSITIVE_J=metric(.97,.66,.77,.7),
            BRIDGE_CE=metric(.99,.68,.79,.72),BRIDGE_SIG=metric(.989,.695,.80,.73))

    def test_source_rule_is_unweighted_dual_gain_and_local_increment(self):
        v=self.records();decision=report.decide(v)
        self.assertTrue(decision['family_expandable'] and decision['local_SIG_increment']['passed'])
        self.assertFalse(decision['uses_target_metrics'] or decision['automatic_seed_expansion'])
        # A higher low score cannot compensate for a failed clean requirement.
        v['BRIDGE_CE']['clean_accuracy']=.97;v['BRIDGE_SIG']['clean_accuracy']=.97
        self.assertFalse(report.decide(v)['family_expandable'])
        for change in (dict(epoch=49),dict(query_read=True),dict(source_count=6300),dict(source_only=False)):
            v=self.records();v['BRIDGE_SIG'].update(change)
            with self.assertRaises(ValueError):report.decide(v)

    def prepare(self,folder):
        row=d.rows()[0];c=d.config(row);output=Path(folder)/row['row_id']/'prediction';output.mkdir(parents=True)
        ids=np.asarray(['unit0','unit1','unit2']);logits=np.eye(3,6,dtype=np.float32);values=logits.argmax(1).astype(np.int64)
        done=dict(status='PREDICTIONS_COMPLETE',config=c,count=3,views=list(d.VIEWS),truth_read=False,query_fit=False,
            all_parameters_frozen=True,head_fit_updates=0)
        d.write(output/'complete.json',done)
        np.savez_compressed(output/'predictions.npz',ids=ids,**{v:values for v in d.VIEWS})
        for view in d.VIEWS:np.savez_compressed(output/(view+'_scores.npz'),ids=ids,logits=logits)
        return row['row_id'],output,ids,done

    def test_all_seven_scores_required_before_truth_and_bad_logits_rejected(self):
        with tempfile.TemporaryDirectory() as temp,patch.object(d,'BASE',Path(temp)),patch.object(evaluate,'COUNT',3):
            row,out,ids,done=self.prepare(temp)
            original=d.read;truth_reads=[]
            def read(path):
                if Path(path)==d.TRUTH:truth_reads.append(path);raise AssertionError('Truth must stay closed')
                return original(path)
            with patch.object(d,'read',side_effect=read),patch.object(evaluate,'observations',return_value=(ids,{})):
                c,actual=evaluate.preflight(row);self.assertTrue(np.array_equal(actual,ids))
                bad=copy.deepcopy(done);bad['views']=list(d.VIEWS[:-1]);d.write(out/'complete.json',bad)
                with self.assertRaises(ValueError):evaluate.score(row)
                d.write(out/'complete.json',done);np.savez_compressed(out/(d.VIEWS[-1]+'_scores.npz'),ids=ids,logits=np.full((3,6),np.nan))
                with self.assertRaises(ValueError):evaluate.score(row)
                self.assertEqual(truth_reads,[])
                self.assertFalse((Path(temp)/row/'scoring').exists())

    def test_prediction_output_is_never_overwritten_or_relaunched(self):
        with tempfile.TemporaryDirectory() as temp,patch.object(d,'BASE',Path(temp)):
            row=d.rows()[0]['row_id'];(Path(temp)/row/'prediction').mkdir(parents=True)
            with patch.object(model,'own_model',side_effect=AssertionError('Do not load CP on existing output')) as loader:
                with self.assertRaises(FileExistsError):evaluate.predict(row)
                loader.assert_not_called()

    def test_unverified_source_mechanism_prevents_query_model_loading(self):
        c=d.config(d.rows()[0])
        bad=dict(status='VERIFIED',updates=11100,main_forward_packets=9031800,head_fit_updates=1,U_truth_read=False)
        with patch.object(d,'read',return_value=bad),patch.object(model,'load_source',side_effect=AssertionError('No query model load')) as loader:
            with self.assertRaises(ValueError):model.own_model(c)
            loader.assert_not_called()
        for row in d.rows():
            config=d.config(row)
            self.assertFalse(config['reused'] or config['baseline_reuse'])
            self.assertEqual(config['bridge'],c['bridge'])
        self.assertEqual(set(d.references()),{'CLEAN','NOMINAL','POSITIVE_J'})

    def test_development_or_other_physical_ID_is_rejected_before_HQ_supervision(self):
        import torch
        from .observer import Observer
        observer=Observer(d.config(d.rows()[0]));observer.allowed={1,2}
        meta=dict(base_index=torch.tensor([1,99]),rx_i=torch.tensor([1,3]),day_i=torch.tensor([1,1]))
        with self.assertRaises(ValueError):observer.track_clean(meta,torch.tensor([0,1]))
        self.assertEqual(observer.clean_seen,set());self.assertEqual(dict(observer.clean_coverage),{})


if __name__=='__main__':unittest.main()
