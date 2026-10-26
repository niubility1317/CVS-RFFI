import unittest
from experiments.pamf_u_history.analyze_vat import scientific_rule


class VATRuleTests(unittest.TestCase):
    def data(self):
        s=dict(calls=6660,seen=1701000,unique_seen=56700,min_visits=30,max_visits=30,VAT_seen=1701000,
            nonzero_direction_fraction=1.,max_relative_radius=.03,probe_parameter_grad_and_buffer_checks=1,
            U_truth_read=False,target_reads=0,model_EMA_updates=0)
        names=('raw','G_raw','practical_low_suburban','G_practical_low_suburban','practical_mid_urban','G_practical_mid_urban','practical_low_urban','G_practical_low_urban')
        rows=[dict(view=v,delta_pp=.1,net_fix=27) for v in names]
        return s,rows

    def test_source_necessary_conditions_not_just_training_completion(self):
        s,rows=self.data();self.assertTrue(scientific_rule(s,rows)['passed'])
        rows[-1]['net_fix']=-1000;rows[-1]['delta_pp']=-1000/270
        self.assertFalse(scientific_rule(s,rows)['passed'])

    def test_coverage_direction_and_radius_are_required(self):
        for key,value in [('VAT_seen',1),('nonzero_direction_fraction',.98),('max_relative_radius',.031),('min_visits',29)]:
            s,rows=self.data();s[key]=value
            self.assertFalse(scientific_rule(s,rows)['passed'])

    def test_missing_view_is_incomplete(self):
        s,rows=self.data();self.assertTrue(scientific_rule(s,rows[:-1])['incomplete'])


if __name__=='__main__':unittest.main()
