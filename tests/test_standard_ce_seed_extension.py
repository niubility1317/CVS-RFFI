import copy
import unittest
import tempfile
from unittest.mock import patch
from pathlib import Path
from experiments.standard_ce_baselines.common import ROOT,read,SEEDS,EXTENSION_SEEDS,EXTENSION_CONFIG,EXTENSION_RUN
from experiments.standard_ce_baselines.dispatch import validate_spec,finish
from experiments.standard_ce_baselines.source import validate_config
from experiments.standard_ce_baselines.combine_seeds import aggregate


class ExtensionTests(unittest.TestCase):
    def test_fixed_52_rows_and_unchanged_training_plan(self):
        spec=read(ROOT/EXTENSION_CONFIG/'launch_spec.json');validate_spec(spec)
        self.assertEqual(len(spec['rows']),52)
        for row in spec['rows']:
            self.assertNotEqual(row['variant'],'native')
            cfg=read(ROOT/EXTENSION_CONFIG/(row['row_id']+'.json'));validate_config(cfg)
            old=read(ROOT/'experiments/standard_ce_baselines/configs'/(row['variant']+'-s2026092701.json'))
            self.assertEqual({k:v for k,v in cfg.items() if k not in ('model_seed','output_root')},
                             {k:v for k,v in old.items() if k not in ('model_seed','output_root')})
        for mutate in ('duplicate','native','oldseed','path','owner'):
            changed=copy.deepcopy(spec)
            if mutate=='duplicate':changed['rows'][-1]=changed['rows'][0]
            if mutate=='native':changed['rows'][0]['variant']='native'
            if mutate=='oldseed':changed['rows'][0]['model_seed']=SEEDS[0]
            if mutate=='path':changed['rows'][0]['source_output']='/outside'
            if mutate=='owner':changed['launch_owner']='other-owner'
            with self.subTest(mutate=mutate),self.assertRaises(ValueError):validate_spec(changed)

    def test_eight_seed_statistics_and_missing_or_duplicate_rejection(self):
        rows=[dict(variant='residual_fusion',model_seed=s,dimension='overall',stratum='ALL',query_count=168000,
                   accuracy=i/10,macro_f1=i/10) for i,s in enumerate(SEEDS+EXTENSION_SEEDS)]
        result=aggregate(rows)
        self.assertEqual(result[0]['actual_seed_count'],8)
        self.assertAlmostEqual(result[0]['accuracy_mean'],.35)
        with self.assertRaises(ValueError):aggregate(rows[:-1])
        with self.assertRaises(ValueError):aggregate(rows+[rows[0]])

    def test_failed_combine_cannot_claim_analyzed(self):
        with tempfile.TemporaryDirectory() as folder:
            spec=dict(run_id=EXTENSION_RUN,runtime_root=folder)
            state=dict(rows={'row':dict(status='SCORED')})
            with patch('experiments.standard_ce_baselines.combine_seeds.combine',side_effect=ValueError('Missing old scores')),patch('experiments.standard_ce_baselines.dispatch.summary'):
                self.assertEqual(finish(spec,state,{}),1)
            saved=read(Path(folder)/'pipeline_state.json')
            self.assertEqual(saved['status'],'FAILED_COMBINE')
            self.assertEqual(saved['rows']['row']['status'],'SCORED')


if __name__=='__main__':unittest.main()
