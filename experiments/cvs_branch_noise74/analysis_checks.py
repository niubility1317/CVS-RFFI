"""P74 pure numeric/source-metadata checks; no model, IQ, checkpoint or remote."""
import copy
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
import numpy as np


def load_analysis():
    path=Path(__file__).with_name('analysis.py');spec=importlib.util.spec_from_file_location('_p74_cpu_analysis',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def logits(pred):
    z=np.zeros((len(pred),6),float);z[np.arange(len(pred)),pred]=3.;return z


def fixture(a,root,n=30):
    y=np.arange(n)%6;ids=np.asarray(['p'+str(i) for i in range(n)]);rx=np.asarray([a.RX[i%5] for i in range(n)]);day=np.arange(n)%3+1
    arrays={}
    for model in a.MODELS:
        clean=y.copy()
        if model=='NoAs':clean[0]=(clean[0]+1)%6
        z=logits(clean);full=np.empty((2,2,n,6));noise=np.empty_like(full);inter=np.empty((2,2,3,4,n,6))
        for draw in (0,1):
            for scene in (0,1):
                pred=y.copy();bad=np.arange(n)%(3 if model=='NoAs' else 4)==0;pred[bad]=(pred[bad]+1)%6
                full[draw,scene]=logits(pred);noise[draw,scene]=logits(y)
                for kind in range(3):
                    for branch in range(4):inter[draw,scene,kind,branch]=noise[draw,scene] if branch==1 else full[draw,scene]
        values=dict(ids=ids,TX=y,RX=rx,day=day,clean=z,clean_zero=np.stack([z]*4),full=full,no_noise_fixed=noise,interventions=inter)
        arrays[model]=values;(root/model).mkdir();np.savez(root/model/'predictions.npz',**values)
    records=[]
    for draw in (0,1):
        for scene in a.SCENES:
            for i in range(n):
                records.append(dict(draw=draw,scene=scene,index=i,physical_id=ids[i],TX=int(y[i]),RX=int(rx[i]),day=int(day[i]),
                    conditions=dict(full=dict(state_end=a.STATES[i%3],receiver_lock_status=a.LOCKS[(i//3)%3],
                        quality_snr_db=12.,signal_to_added_noise_db=(-2.,3.,8.,15.)[i%4],actual_waveform_quality_db=11.,agc_limited=False))))
    (root/'panel_metadata.json').write_text(json.dumps(dict(full_records=records)),encoding='utf8')
    (root/'complete.json').write_text(json.dumps(dict(status='SOURCE_DIAGNOSTIC_COMPLETE',source_only=True,target_access=False,
        training_updates=0,head_fit_updates=0)),encoding='utf8')
    return arrays


def checks():
    a=load_analysis()
    class Checks(unittest.TestCase):
        def test_exact_paired_repairs_errors_and_margin_scale(self):
            y=np.arange(6);base=logits(np.asarray([0,0,2,3,4,5]));new=logits(np.asarray([0,1,1,3,4,5]))
            result=a.paired(base,new,y,np.asarray([True]*5+[False]))
            self.assertEqual((result['repairs'],result['new_errors'],result['net_repairs']),(1,1,0))
            self.assertEqual((result['cohort_repairs'],result['cohort_new_errors']),(1,1))
            np.testing.assert_allclose(a.margin30(base,y),[.1,-.1,.1,.1,.1,.1])
            self.assertAlmostEqual(result['mean_margin_over30_delta'],0.)
            perfect=a.metrics(logits(y),y,np.asarray([1,1,3,3,4,4]),np.ones(6,bool))
            self.assertEqual(perfect['accuracy'],1.);self.assertEqual(perfect['macro_f1'],1.)
            self.assertEqual(perfect['worst_RX_IDs'],[1,3,4])

        def test_empty_strata_and_empty_cohort_remain_undefined(self):
            v=a.metrics(np.empty((0,6)),np.asarray([],int),np.asarray([],int),np.asarray([],bool))
            self.assertEqual(v['count'],0);self.assertIsNone(v['accuracy']);self.assertIsNone(v['macro_f1'])
            self.assertIsNone(v['worst_RX']);self.assertIsNone(v['satellite_retention'])
            p=a.paired(np.empty((0,6)),np.empty((0,6)),np.asarray([],int),np.asarray([],bool))
            self.assertEqual(p['repairs'],0);self.assertIsNone(p['accuracy_delta_pp'])
            v=a.metrics(logits(np.arange(6)),np.arange(6),np.ones(6,int),np.zeros(6,bool))
            self.assertIsNone(v['satellite_retention'])

        def test_fixed_NoAs30_cohort_and_two_draws_not_model_seeds(self):
            with tempfile.TemporaryDirectory() as temp:
                p=Path(temp);fixture(a,p);arrays,physics,_,_=a.load_inputs(p,30);data=a.analyze_arrays(arrays,physics)
                self.assertEqual(data['fixed_NoAs30_clean_correct_count'],29)
                value=data['overall'][('As',a.SCENES[0],-1,'full','ALL')]
                self.assertEqual(value['fixed_NoAs30_clean_correct_count'],58)
                self.assertEqual(value['count'],60)
                rules=a.attribution_rules(data);self.assertEqual(rules['model_seed_count'],1)
                self.assertEqual(rules['channel_draw_count'],2);self.assertTrue(rules['channel_draws_are_not_model_seeds'])
                self.assertFalse(rules['automatic_training']);self.assertFalse(rules['automatic_E50'])

        def test_source_only_complete_shapes_and_physical_alignment(self):
            with tempfile.TemporaryDirectory() as temp:
                p=Path(temp);arrays=fixture(a,p);a.load_inputs(p,30)
                c=json.loads((p/'complete.json').read_text());c['target_access']=True
                (p/'complete.json').write_text(json.dumps(c))
                with self.assertRaises(ValueError):a.load_inputs(p,30)
                c['target_access']=False;(p/'complete.json').write_text(json.dumps(c))
                m=json.loads((p/'panel_metadata.json').read_text());m['full_records'][0]['physical_id']='wrong'
                (p/'panel_metadata.json').write_text(json.dumps(m))
                with self.assertRaises(ValueError):a.load_inputs(p,30)
                m['full_records'][0]['physical_id']='p0';(p/'panel_metadata.json').write_text(json.dumps(m))
                bad=dict(arrays['As']);bad['interventions']=bad['interventions'][:,:,:,:3]
                np.savez(p/'As/predictions.npz',**bad)
                with self.assertRaises(ValueError):a.load_inputs(p,30)

        def test_noise_rule_requires_all_four_conditions_and_positive_whole_gain(self):
            with tempfile.TemporaryDirectory() as temp:
                p=Path(temp);fixture(a,p);arrays,physics,_,_=a.load_inputs(p,30);data=a.analyze_arrays(arrays,physics)
                rules=a.attribution_rules(data);pa=next(r for r in rules['branches'] if r['branch']=='PA')
                self.assertTrue(pa['noise_specific_local_evidence']);self.assertTrue(pa['lightweight_masking_worth_research'])
                bad=copy.deepcopy(data);bad['overall'][('As',a.SCENES[1],1,'no_noise_fixed','ALL')]['effect']['accuracy_delta_pp']=0.
                self.assertFalse(next(r for r in a.attribution_rules(bad)['branches'] if r['branch']=='PA')['noise_specific_local_evidence'])
                bad=copy.deepcopy(data)
                next(r for r in bad['clean'] if r['model']=='As' and r['condition']=='clean_zero' and r['branch']=='PA')['clean_zero_cost_pp']=.10001
                self.assertFalse(next(r for r in a.attribution_rules(bad)['branches'] if r['branch']=='PA')['lightweight_masking_worth_research'])

        def test_one_fixed_source_branch_not_best_draw_selection(self):
            with tempfile.TemporaryDirectory() as temp:
                p=Path(temp);fixture(a,p);arrays,physics,_,_=a.load_inputs(p,30);data=a.analyze_arrays(arrays,physics)
                choice=a.attribution_rules(data)['descriptive_fixed_swap']
                self.assertEqual(choice['branch'],'PA');self.assertTrue(choice['source_analysis_selected'])
                self.assertFalse(choice['per_draw_selection']);self.assertTrue(choice['not_a_training_candidate'])

        def test_three_numeric_figures_png_pdf_and_tables(self):
            with tempfile.TemporaryDirectory() as temp:
                p=Path(temp);fixture(a,p);arrays,physics,_,_=a.load_inputs(p,30);data=a.analyze_arrays(arrays,physics);rules=a.attribution_rules(data)
                out=p/'analysis';out.mkdir();files=a.plots(out,data,rules)
                self.assertEqual(len(files),6);self.assertTrue(all(Path(f).is_file() and Path(f).stat().st_size>100 for f in files))
                a.markdown(out,data,rules,files);self.assertIn('不是跨seed复现',(out/'readout.md').read_text(encoding='utf8'))
                a.csv_write(out/'paired.csv',data['paired']);self.assertTrue((out/'paired.csv').is_file())

        def test_actual_added_noise_boundaries_and_all_strata(self):
            self.assertEqual([a.noise_bin(x) for x in (-2.,3.,8.,15.,None,float('nan'))],['<3','3-8','8-15','>=15','MISSING','MISSING'])
            with tempfile.TemporaryDirectory() as temp:
                p=Path(temp);fixture(a,p);arrays,physics,_,_=a.load_inputs(p,30)
                strata=a.stratification(physics,a.SCENES[0],(0,1),30,arrays['NoAs'])
                dimensions={k for k,_,_ in strata};self.assertEqual(dimensions,{'overall','state','lock','noise_bin','state_x_lock','RX','day'})
                self.assertEqual(sum(int(mask.sum()) for dim,_,mask in strata if dim=='state_x_lock'),60)
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Checks))
    if not result.wasSuccessful():raise SystemExit(1)


if __name__=='__main__':checks()
