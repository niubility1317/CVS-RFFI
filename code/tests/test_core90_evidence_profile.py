import json
import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch
from cvsrffi.core90_evidence_profile import core90_arguments, load_profile
from SSDG.train_ssdg import build_arg_parser


class ProfileTests(unittest.TestCase):
    def arguments(self,variant='H0'):
        return core90_arguments('dataset.pkl','contract.json','output',variant,
                                source_rxs='0,1',source_days='0,1',target_rxs='2',target_days='2')

    def test_original_core90_and_protocol(self):
        profile=load_profile()
        self.assertIn('phase1_adv3_mechanism32_queue_20260701',profile['reference_parser'])
        self.assertEqual(profile['historical_parser_defaults']['labeled_ratio'],.10)
        args=build_arg_parser().parse_args(self.arguments())
        self.assertEqual((args.epochs,args.label_epochs,args.pseudo_epochs),(200,130,70))
        self.assertEqual((args.sat_cons_start_epoch,args.lambda_sat_cls,args.lambda_sat_cons),(80,.68,0))
        self.assertEqual((args.proxy_unknown_core_quantile,args.proxy_unknown_accept_quantile,args.proxy_unknown_vaccept_cvar_alpha,args.proxy_unknown_core_accept_weight),(.9,.85,.3,.45))
        self.assertEqual((args.labeled_ratio,args.unlabeled_ratio,args.source_val_ratio),(.07,.63,.3))
        self.assertEqual(args.phase1_source_role_protocol,'legacy_l_u_v')
        self.assertEqual(args.phase1_terminal_policy,'core90_research')
        self.assertEqual(args.checkpoint_selection,'final_only')
        self.assertEqual(args.test_eval_policy,'final_only')
        self.assertEqual(args.best_metric,'clean_val_tx')
        self.assertFalse(args.enable_joint_safe_guard)
        self.assertTrue(args.phase1_source_val_selection_only)
        self.assertTrue(args.from_scratch)
        self.assertFalse(args.freeze_backbone)
        self.assertEqual((args.baseline_ckpt,args.teacher_ckpt),('',''))
        self.assertTrue(args.use_ema_teacher)
        self.assertEqual((args.model_variant,args.branch_ablation,args.domain_branch_ablation),('lite_d','no_dac','no_stats'))
        self.assertEqual((args.mixstyle_p,args.mixstyle_strength),(.18,.70))
        self.assertEqual(args.wisig_train_rxs,'0,1')
        self.assertEqual(args.evidence_config,'')
        for name in ('use_muse_ssdg','fasttrust_rc4','sat_anchor_ssl','use_crra','phase1_v2_hard_gates'):
            self.assertFalse(getattr(args,name),name)
        for name in ('endpoint_require_artifact_on_export','tail_stop_blocks_final',
                     'phase1_v2_guard_blocks_final','u_direct_idle_blocks_promotion',
                     'tail_safety_training_stop_enabled'):
            self.assertFalse(getattr(args,name),name)
        self.assertEqual(args.evidence_data_contract,'contract.json')
        self.assertEqual(args.phase1_source_role_protocol,'legacy_l_u_v')
        for name in profile['later_addon_overrides']:
            if name.startswith('lambda_'):
                self.assertEqual(getattr(args,name),0,name)

    def test_variants_and_reproducibility(self):
        from cvsrffi.evidence_head import EvidenceConfig
        for i in range(1,6):
            argv=self.arguments(f'H{i}')
            self.assertEqual(argv,self.arguments(f'H{i}'))
            args=build_arg_parser().parse_args(argv)
            config=EvidenceConfig.parse(args.evidence_config)
            self.assertEqual(config.covariance_rank,0 if i==1 else 4)
            self.assertEqual(config.variant,f'H{i}')

    def test_unsupported_is_fatal(self):
        profile=load_profile()
        profile['historical_parser_defaults']['unsupported_original_argument']=1
        with patch('cvsrffi.core90_evidence_profile.load_profile',return_value=profile):
            with self.assertRaisesRegex(ValueError,'unsupported_original_argument'):
                self.arguments()

    def test_research_completion_still_requires_artifacts_and_protocol(self):
        from SSDG.train_ssdg import _resolve_phase1_terminal_status, _core90_research_export_complete
        flags=dict(tail_stopped=False,export_failed=False,final_blocked=False,
                   selected_checkpoint_exists=True,heldout_eval_status='COMPLETE',
                   p0_mechanisms_ready=False,p1_mechanisms_ready=False,
                   endpoint_export_ready=False,mechanism_gates_required=False,
                   endpoint_export_required=False)
        self.assertEqual(_resolve_phase1_terminal_status(**flags),'COMPLETE')
        for key,value,expected in (
            ('export_failed',True,'FAILED_EXPORT'),
            ('selected_checkpoint_exists',False,'NO_SAFE_CHECKPOINT'),
            ('heldout_eval_status','FAILED','HELDOUT_EVAL_INCOMPLETE'),
            ('final_blocked',True,'NON_PROMOTABLE_GUARD_BLOCKED'),
            ('mechanism_gates_required',True,'NON_PROMOTABLE_P0_DISABLED')):
            self.assertEqual(_resolve_phase1_terminal_status(**{**flags,key:value}),expected)
        with tempfile.TemporaryDirectory() as folder:
            pt, js=Path(folder)/'prototype.pt',Path(folder)/'prototype.json'
            status=dict(status='COMPLETE',prototype_path=str(pt),prototype_json_path=str(js))
            self.assertFalse(_core90_research_export_complete(status))
            pt.write_bytes(b'fixture')
            self.assertFalse(_core90_research_export_complete(status))
            js.write_text('{}',encoding='utf-8')
            self.assertTrue(_core90_research_export_complete(status))
            self.assertFalse(_core90_research_export_complete({**status,'status':'FAILED'}))


if __name__=='__main__':
    unittest.main()
