import unittest
import gzip
import json
import tempfile
from pathlib import Path
from unittest.mock import patch
from experiments.pamf_repl_reference_comparison import comparison_plan,dominance_stats,analyze


class ReferenceComparisonTests(unittest.TestCase):
    def test_all_reference_modes_pair_by_physical_id(self):
        import numpy as np
        def write(path,value):
            path.parent.mkdir(parents=True,exist_ok=True)
            path.write_text(json.dumps(value),encoding='utf-8')
        with tempfile.TemporaryDirectory() as tmp:
            project=Path(tmp);roots={m:project/'runs'/m for m in ('dual','full','CE','EMA')}
            truth={'a':{'label':0,'receiver':'r1','day':'d1'},'b':{'label':1,'receiver':'r2','day':'d2'}}
            handles={};predictions={}
            for setting,protocol in (('ten_class','ManyRx'),('source_rx_6','m6-natural')):
                tasks={}
                for task in ('cross_rx','cross_both'):
                    path=project/f'{setting}_{task}.json';write(path,truth);tasks[task]={'truth':str(path)}
                handles[protocol]={'source_contract':f'{protocol}_contract','tests':tasks}
            for mode,root in roots.items():
                sources={}
                for setting,protocol in (('ten_class','ManyRx'),('source_rx_6','m6-natural')):
                    for weight in (3.,6.):
                        for cap in ((None,) if mode=='dual' else (.25,.5)):
                            rid=f'{setting}-{mode}-{weight}-{cap}';source=root/'F'/rid/'source';pred=root/'readouts'/rid/'predictions';score=root/'readouts'/rid/'scores';u=root/'readouts'/rid/'U_diagnostic'
                            recipe={'weight':weight}
                            if cap is not None:recipe['reference_cap']=cap
                            cfg={'row_id':rid,'model_seed':2026092701,'protocol_id':protocol,'tuning_recipe':recipe}
                            write(source/'complete.json',{'config':cfg,'status':'SOURCE_FROZEN','epoch':50,'updates':11100,'target_access':False,'U_truth_read':False})
                            right=mode in ('full','EMA');confusion=[[1,0],[0,1]] if right else [[0,1],[0,1]]
                            score_rows=[{'test':task,'readout':'raw','view':'clean','dimension':'overall','count':2,'confusion':confusion} for task in ('cross_rx','cross_both')]
                            write(score/'complete.json',{'status':'SCORED_COMPLETE'});write(score/'scores.json',{'truth_last':True,'results':score_rows})
                            write(u/'complete.json',{'status':'SCORED_U_DIAGNOSTIC','truth_last':True,'source_epoch':50})
                            write(u/'scores.json',{'training_or_selection_use':False,'overall':{'seen':2,'accepted':2,'accepted_correct':2},'unique_U_total':2,'unique_accepted':2})
                            predictions[str(pred)]=(np.array(['a','b']),np.array([0,1] if right else [1,1]))
                            sources[rid]={'source_output':str(source),'predictions':str(pred),'scores':str(score)}
                release=project/'releases'/(mode+'-release')
                write(root/'pipeline_state.json',{'status':'ANALYZED'});write(root/'sources.json',sources)
                write(release/'runtime.json',{'context':str(release/'context.json')});write(release/'context.json',{'handles':handles})
            with patch('experiments.pamf_repl_dual_ce.analyze_segment.analyze',return_value={}), \
                 patch('experiments.pamf_repl_reference_comparison.dominance_stats',return_value={}), \
                 patch('experiments.pamf_repl_weight.analyze_paired.prediction',side_effect=lambda folder,task:predictions[str(folder)]):
                result=analyze(roots['dual'],roots['full'],roots['CE'],roots['EMA'])
            self.assertEqual(len(result['cells']),28)
            self.assertEqual(len(result['comparisons']),48)
            ce_full=next(r for r in result['comparisons'] if r['comparison']=='CE_vs_full')
            self.assertEqual(ce_full['delta_pp'],-50.)
            self.assertEqual(ce_full['combined']['damage'],2)
            first=next(iter(predictions));predictions[first]=(np.array(['a','wrong']),np.array([0,1]))
            with patch('experiments.pamf_repl_dual_ce.analyze_segment.analyze',return_value={}), \
                 patch('experiments.pamf_repl_reference_comparison.dominance_stats',return_value={}), \
                 patch('experiments.pamf_repl_weight.analyze_paired.prediction',side_effect=lambda folder,task:predictions[str(folder)]):
                with self.assertRaisesRegex(ValueError,'physical IDs'):
                    analyze(roots['dual'],roots['full'],roots['CE'],roots['EMA'])

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
