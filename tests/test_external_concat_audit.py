import copy,unittest
from experiments.phase1_external_dg_aug.concat_audit import audit_stage,audit_selected

class ConcatAuditTests(unittest.TestCase):
    def setUp(self):
        self.c=dict(satellite_train_augmentation=True,concat_sat_ce_only=True,lambda_sat_cls=.68,
            lambda_sat_cons=0.,sat_ce_start=1,default_head_joint_training=True,separate_head_training=False,
            U_s_use=False,pl_weight=0.)
        self.stage=dict(stage=1,log_intervals=1,updates=2,labeled_exposure=12,satellite_examples=12,
            satellite_applied_examples=6,pl_seen=0,pl_accepted=0,optimizer_steps=2)
        self.r=dict(loss=1.+.68*2.+.3,identity_ce=1.,satellite_ce=2.,native_penalty=.3,pl_loss=0.,
            satellite_weight=.68,concat_sat_ce_only=True,target_access=False,pseudo_label_truth_access=False,
            U_s_use=False,updates=2,physical_source_exposure_total=12,satellite_samples_total=12,
            satellite_applied_samples_total=6,pl_seen_total=0,pl_accepted_total=0,optimizer_steps_total=2,
            satellite_scenarios=['clean_duplicate','leo_clear_weak'])
    def test_valid_executed_loss_and_selected_counts(self):
        self.assertEqual(audit_stage(self.c,self.stage,[self.r])['status'],'PASS')
        cost=dict(satellite_examples=12,satellite_applied_examples=6,pl_seen=0,pl_accepted=0,
            selected_training_telemetry=dict(satellite_examples=6,satellite_applied_examples=3,pl_seen=0,pl_accepted=0))
        self.assertEqual(audit_selected(self.c,cost)['status'],'PASS')
    def test_detects_disabled_loss_despite_positive_exposure_telemetry(self):
        bad=copy.deepcopy(self.r);bad['loss']-=.68*bad['satellite_ce']
        self.assertEqual(audit_stage(self.c,self.stage,[bad])['status'],'FAIL')
    def test_detects_selected_unaugmented_and_U_access(self):
        cost=dict(satellite_examples=12,satellite_applied_examples=6,pl_seen=0,pl_accepted=0,
            selected_training_telemetry=dict(satellite_examples=6,satellite_applied_examples=0,pl_seen=0,pl_accepted=0))
        self.assertEqual(audit_selected(self.c,cost)['status'],'FAIL')
        bad=copy.deepcopy(self.r);bad['pl_seen_total']=12
        self.assertEqual(audit_stage(self.c,self.stage,[bad])['status'],'FAIL')

if __name__=='__main__':unittest.main()
