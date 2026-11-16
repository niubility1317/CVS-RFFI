import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from experiments.pamf_repl_factorial_effects import contrast_plan, summarize_effects, analyze


class FactorialEffectTests(unittest.TestCase):
    def test_complete_matrix_is_scored_and_paired_on_exact_ids(self):
        import numpy as np
        def write(path,value):
            path.parent.mkdir(parents=True,exist_ok=True)
            path.write_text(json.dumps(value),encoding='utf-8')
        with tempfile.TemporaryDirectory() as tmp:
            project=Path(tmp);root=project/'runs'/'factorial';release=project/'releases'/'factorial-release'
            truth={'a':{'label':0,'receiver':'r1','day':'d1'},'b':{'label':1,'receiver':'r2','day':'d2'}}
            handles={};sources={};fake_predictions={}
            for setting,protocol in (('ten_class','ManyRx'),('source_rx_6','m6-natural')):
                truth_paths={}
                for task in ('cross_rx','cross_both'):
                    path=project/f'{setting}_{task}.json';write(path,truth);truth_paths[task]={'truth':str(path)}
                handles[protocol]={'tests':truth_paths}
                for weight in (3.,6.):
                    for a in (0,1):
                        for b in (0,1):
                            rid=f'{setting}-w{int(weight)}-a{a}-b{b}';source=root/'F'/rid/'source';pred=root/'readouts'/rid/'predictions';score=root/'readouts'/rid/'scores';u=root/'readouts'/rid/'U_diagnostic'
                            cfg={'row_id':rid,'model_seed':2026092701,'protocol_id':protocol,
                                 'tuning_recipe':{'weight':weight,'shallow_factor':float(a),'extrap_coefficient':.25*b}}
                            write(source/'complete.json',{'config':cfg});write(score/'complete.json',{'status':'SCORED_COMPLETE'})
                            confusion=[[1,0],[0,1]] if a else [[0,1],[0,1]]
                            records=[{'test':task,'readout':'raw','view':'clean','dimension':'overall','count':2,'confusion':confusion}
                                     for task in ('cross_rx','cross_both')]
                            write(score/'scores.json',{'truth_last':True,'results':records})
                            write(u/'complete.json',{'status':'SCORED_U_DIAGNOSTIC','truth_last':True,'source_epoch':50})
                            write(u/'scores.json',{'training_or_selection_use':False,'overall':{'seen':2,'accepted':2,'accepted_correct':2},
                                                   'unique_U_total':2,'unique_seen':2,'unique_accepted':2})
                            fake_predictions[str(pred)]=(np.array(['a','b']),np.array([0,1] if a else [1,1]))
                            sources[rid]={'source_output':str(source),'predictions':str(pred),'scores':str(score)}
            write(root/'pipeline_state.json',{'status':'ANALYZED'});write(root/'sources.json',sources)
            write(release/'runtime.json',{'context':str(release/'context.json')});write(release/'context.json',{'handles':handles})
            with patch('experiments.pamf_repl_factorial_effects.source_metrics',return_value={'source_val_E50':90.}), \
                 patch('experiments.pamf_repl_weight.analyze_paired.prediction',side_effect=lambda folder,task:fake_predictions[str(folder)]):
                result=analyze(root,'shallow_extrap')
            for setting in ('ten_class','source_rx_6'):
                effects=result['settings'][setting]['effects']['clean_pp']['by_weight']
                self.assertEqual(effects['3.0']['A_at_B0'],50.)
                self.assertEqual(effects['6.0']['AB_interaction'],0.)
                self.assertEqual(len(result['settings'][setting]['contrasts']),12)
            first=next(iter(fake_predictions));fake_predictions[first]=(np.array(['a','wrong']),np.array([0,1]))
            with patch('experiments.pamf_repl_factorial_effects.source_metrics',return_value={'source_val_E50':90.}), \
                 patch('experiments.pamf_repl_weight.analyze_paired.prediction',side_effect=lambda folder,task:fake_predictions[str(folder)]):
                with self.assertRaisesRegex(ValueError,'physical IDs'):
                    analyze(root,'shallow_extrap')

    def test_simple_effects_and_interaction_are_not_conflated(self):
        cells = {(w,a,b): 80+2*(w==6)+3*a+5*b+7*a*b
                 for w in (3.,6.) for a in (0,1) for b in (0,1)}
        result = summarize_effects(cells)
        self.assertEqual(result['by_weight']['3.0']['A_at_B0'],3)
        self.assertEqual(result['by_weight']['3.0']['A_at_B1'],10)
        self.assertEqual(result['by_weight']['3.0']['B_at_A0'],5)
        self.assertEqual(result['by_weight']['3.0']['B_at_A1'],12)
        self.assertEqual(result['by_weight']['3.0']['AB_interaction'],7)
        self.assertEqual(result['weight_effects']['A1_B1'],2)
        self.assertEqual(len(contrast_plan()),12)

    def test_missing_cell_is_an_error(self):
        cells = {(w,a,b):1. for w in (3.,6.) for a in (0,1) for b in (0,1)}
        del cells[(6.,1,1)]
        with self.assertRaises(ValueError):
            summarize_effects(cells)


if __name__ == '__main__':
    unittest.main()
