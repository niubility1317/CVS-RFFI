import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
from experiments.pamf_repl_weight.analyze_paired import analyze


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value),encoding='utf-8')


def predictions(path,values):
    write(path/'complete.json',dict(status='ALL_PREDICTIONS_COMPLETE',truth_read=False))
    for task in ('cross_rx','cross_both'):
        folder=path/task;folder.mkdir(parents=True)
        np.savez(folder/'predictions.npz',ids=np.array(['a','b']),raw__clean=np.array(values))


def score(path):
    write(path/'complete.json',dict(status='SCORED_COMPLETE'))
    write(path/'scores.json',dict(truth_last=True))


class PairedSupervisedTests(unittest.TestCase):
    def test_matched_supervised_and_existing_controls(self):
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp);root=base/'runs'/'factor';sup=base/'runs'/'supervised';release=base/'releases'/'factor-release'
            sources={};controls={};sup_sources={};handles={}
            for setting,protocol in (('ten_class','ManyRx'),('source_rx_6','m6-natural')):
                contract=protocol+'_source';rid=setting+'-candidate';spid=setting+'-tri-label-only-permuted-s2026092701'
                pred=root/rid/'predictions';scored=root/rid/'scores';predictions(pred,[0,1]);score(scored)
                sources[rid]={'predictions':str(pred),'scores':str(scored)}
                write(root/'configs'/('predict__'+rid+'.json'),dict(protocol_id=protocol,source_contract=contract))
                controls[setting]={}
                for kind in ('pl','tri'):
                    p=root/rid/kind/'predictions';s=root/rid/kind/'scores';predictions(p,[1,1]);score(s)
                    controls[setting][kind]=dict(predictions=str(p),scores=str(s))
                source=sup/spid/'source';p=sup/spid/'predictions';s=sup/spid/'scores';predictions(p,[1,1]);score(s)
                write(source/'complete.json',dict(status='SOURCE_FROZEN',epoch=50,updates=11100,
                    target_access=False,U_truth_read=False,config=dict(model_seed=2026092701,U_use=False,
                    regime='matched_sat_supervised',protocol_id=protocol,source_contract=contract)))
                sup_sources[spid]=dict(source_output=str(source),predictions=str(p),scores=str(s))
                truth=base/(protocol+'_truth.json');write(truth,{'a':{'label':0,'receiver':'r1','day':'d1'},
                    'b':{'label':1,'receiver':'r2','day':'d2'}})
                handles[protocol]={'tests':{t:{'truth':str(truth)} for t in ('cross_rx','cross_both')}}
            write(root/'pipeline_state.json',dict(status='ANALYZED'));write(root/'sources.json',sources);write(root/'controls.json',controls)
            write(sup/'pipeline_state.json',dict(status='ANALYZED'));write(sup/'sources.json',sup_sources)
            write(release/'runtime.json',dict(context=str(release/'context.json')));write(release/'context.json',dict(handles=handles))
            rows=analyze(root,sup)['rows']
            self.assertEqual(len(rows),2)
            for row in rows:
                self.assertEqual(set(row['controls']),{'tri','pl','supervised'})
                self.assertEqual(row['controls']['supervised']['combined']['gain'],2)
                self.assertEqual(row['controls']['supervised']['combined']['damage'],0)
            model=Path(next(iter(sup_sources.values()))['source_output'])/'complete.json'
            data=json.loads(model.read_text());data['config']['U_use']=True;write(model,data)
            with self.assertRaisesRegex(ValueError,'label-only'):analyze(root,sup)


if __name__=='__main__':unittest.main()
