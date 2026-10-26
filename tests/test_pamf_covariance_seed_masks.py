import unittest
from copy import deepcopy
from unittest.mock import patch
from experiments.pamf_u_history.covariance_seed_masks import SEEDS,validate_context


class SeedMaskTests(unittest.TestCase):
    def context(self):
        sources={k:'/'+k for k in ('F','AF','BASE','FOLD')}
        return dict(project='/p',python='/python',frozen_sources=sources,channel_code='/physics',source_cache='/cache',source_contract='/contract',U_channel_cache='/H',seed01_masks='/old01',
            cases=[dict(model_seed=s,baseline_config=dict(model_seed=s,paper_method='SIG1-PL',source_cache='/cache',source_contract='/contract',frozen_sources=sources)) for s in SEEDS])

    @patch('experiments.pamf_channel_pl.sig1_matched_baselines.validate_config',lambda c:None)
    def test_model_seed_cannot_take_other_seed_config(self):
        c=self.context();validate_context(c)
        bad=deepcopy(c);bad['cases'][1]['baseline_config']['model_seed']=SEEDS[0]
        with self.assertRaises(ValueError):validate_context(bad)
        bad=deepcopy(c);bad['cases'][2]['model_seed']=SEEDS[1]
        with self.assertRaises(ValueError):validate_context(bad)
        bad=deepcopy(c);bad['cases'].append(deepcopy(bad['cases'][0]))
        with self.assertRaises(ValueError):validate_context(bad)

    @patch('experiments.pamf_channel_pl.sig1_matched_baselines.validate_config',lambda c:None)
    def test_source_role_handles_cannot_change_or_add_G(self):
        c=self.context();bad=deepcopy(c);bad['cases'][1]['baseline_config']['source_cache']='/other'
        with self.assertRaises(ValueError):validate_context(bad)
        bad=deepcopy(c);bad['cases'][1]['baseline_config']['source_G']='/weight'
        with self.assertRaises(ValueError):validate_context(bad)
        bad=deepcopy(c);bad['frozen_sources']['G']='/G'
        with self.assertRaises(ValueError):validate_context(bad)
