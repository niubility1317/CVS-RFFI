import csv
from itertools import product
from pathlib import Path
import tempfile
import unittest
from experiments.cric_dura_comparison.io import read,write
from experiments.cric_dura_comparison.analyze_core import run,VIEWS,verified_metrics
from experiments.cric_dura_comparison.core_groups import SEEDS,switches


def metric(diag):
    cm=[[diag,50-diag],[50-diag,diag]]
    return dict(count=100,accuracy=diag/50,macro_f1=diag/50,confusion=cm)


class CoreAnalysisTests(unittest.TestCase):
    def fixture(self,root):
        cells=[]
        for code in product('01',repeat=3):
            bits=''.join(code);a,b,c=map(int,code)
            for index,seed in enumerate(SEEDS):
                folder=root/(bits+'-s'+str(seed));rows=[];pairs=[]
                base=20+2*a+3*b+4*c+a*b+index
                for reader,offset in (('raw',0),('DURA-L',1),('DURA-LU',2)):
                    for view in VIEWS:
                        row=dict(readout=reader,test='main',view=view,dimension='overall',stratum='ALL',**metric(base+offset))
                        rows.append(row)
                        if offset:pairs.append(dict(readout=reader,test='main',view=view,dimension='overall',stratum='ALL',
                            count=100,repairs=2*offset,damage=0,net=2*offset))
                write(folder/'scores.json',dict(status='SCORED',run_id='fixture',row_id=folder.name,
                    model_seed=seed,truth_last=True,target_feedback_forbidden=True,results=rows,paired_raw=pairs))
                write(folder/'score-complete.json',dict(status='SCORED_COMPLETE'))
                write(folder/'source-complete.json',dict(epoch=50,updates=11100,seconds=10,
                    config={'core_mechanisms':switches(bits)}))
                record=dict(run_id='fixture',row_id=folder.name,model_seed=seed,
                    scores=str(folder/'scores.json'),completion=str(folder/'score-complete.json'))
                cells.append(dict(bits=bits,model_seed=seed,source=dict(new_F=True,completion=str(folder/'source-complete.json')),
                    readouts={r:dict(record,readout=r) for r in ('raw','DURA-L','DURA-LU')}))
        return dict(cells=cells,num_classes=2,output_root=str(root/'results'),main_table=['111','011','101','110'])
    def test_reconstructs_CM_and_known_factorial_effects(self):
        with tempfile.TemporaryDirectory() as folder:
            c=self.fixture(Path(folder));run(c);out=Path(c['output_root'])
            self.assertEqual(read(out/'complete.json')['cells'],24)
            with (out/'core_paired_effects.csv').open(encoding='utf-8',newline='') as f:
                found={r['effect']:r for r in csv.DictReader(f) if r['readout']=='raw' and r['view']=='clean' and r['metric']=='accuracy'}
            for name,value in {'A_at_other_groups_on':6.,'A_factorial_average':5.,
                    'B_factorial_average':7.,'C_factorial_average':8.,'AB_interaction_average':2.,
                    'AC_interaction_average':0.,'BC_interaction_average':0.,'ABC_interaction':0.}.items():
                self.assertAlmostEqual(float(found[name]['mean']),value,places=10)
                self.assertAlmostEqual(float(found[name]['sd']),0.,places=10)
    def test_incomplete_seed_rejected_before_outputs(self):
        with tempfile.TemporaryDirectory() as folder:
            c=self.fixture(Path(folder));c['cells'].pop()
            with self.assertRaisesRegex(ValueError,'factorial'):run(c)
            self.assertFalse(Path(c['output_root']).exists())
    def test_invalid_CM_score_is_rejected(self):
        row=metric(40);row['accuracy']=.9
        with self.assertRaisesRegex(ValueError,'reconstruction'):verified_metrics(row,2)


if __name__=='__main__':unittest.main()
