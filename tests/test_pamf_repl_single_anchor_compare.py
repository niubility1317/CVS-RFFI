import json
import tempfile
import unittest
from pathlib import Path
from tests.test_pamf_repl_paired_supervised import write,predictions
from experiments.pamf_repl_dual_ce.analyze_single_anchor import analyze


def score(path,correct):
    write(path/'complete.json',dict(status='SCORED_COMPLETE'))
    confusion=[[1,0],[0,1]] if correct else [[0,1],[0,1]]
    write(path/'scores.json',dict(truth_last=True,results=[dict(test=t,readout='raw',view='clean',
        dimension='overall',confusion=confusion) for t in ('cross_rx','cross_both')]))


class SingleAnchorComparisonTests(unittest.TestCase):
    def test_four_matched_comparisons_and_contract_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp);roots={x:base/'runs'/x for x in ('dual','single6','single3')};handles={}
            for kind,root in roots.items():
                sources={}
                for setting,protocol in (('ten_class','ManyRx'),('source_rx_6','m6-natural')):
                    for w in ((3,6) if kind=='dual' else ((6,) if kind=='single6' else (3,))):
                        recipe=('dual_ce'+str(w)) if kind=='dual' else ('step_guard' if w==6 else 'step_weight3')
                        rid=f'{setting}-repl-{recipe}-s2026092701';folder=root/rid;source=folder/'source';pred=folder/'predictions';scores=folder/'scores'
                        write(source/'complete.json',dict(status='SOURCE_FROZEN',epoch=50,updates=11100,target_access=False,
                            U_truth_read=False,config=dict(row_id=rid,model_seed=2026092701,protocol_id=protocol,source_contract=protocol)))
                        predictions(pred,[0,1] if kind=='dual' else [1,1]);score(scores,kind=='dual')
                        sources[rid]=dict(source_output=str(source),predictions=str(pred),scores=str(scores))
                    truth=base/(protocol+'_truth.json');write(truth,{'a':{'label':0,'receiver':'r1','day':'d1'},
                        'b':{'label':1,'receiver':'r2','day':'d2'}})
                    handles[protocol]={'tests':{t:{'truth':str(truth)} for t in ('cross_rx','cross_both')}}
                write(root/'pipeline_state.json',dict(status='ANALYZED'));write(root/'sources.json',sources)
            release=base/'releases'/'dual-release';write(release/'runtime.json',dict(context=str(release/'context.json')))
            write(release/'context.json',dict(handles=handles))
            result=analyze(roots['dual'],roots['single6'],roots['single3'])
            self.assertEqual(len(result['rows']),4)
            self.assertTrue(all(row['delta_pp']==50 and row['combined']['gain']==2 for row in result['rows']))
            path=next((roots['single6']).glob('*/source/complete.json'))
            data=json.loads(path.read_text());data['config']['source_contract']='wrong';write(path,data)
            with self.assertRaisesRegex(ValueError,'contract'):analyze(roots['dual'],roots['single6'],roots['single3'])


if __name__=='__main__':unittest.main()
