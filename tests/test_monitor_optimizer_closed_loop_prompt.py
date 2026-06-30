try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10 in ssr-gpu
    import tomli as tomllib
import unittest
from pathlib import Path


AUTOMATION_TOML = Path(
    "E:/codex/home/automations/"
    "cv-sincnet-n607-monitor-optimizer-v4-2/automation.toml"
)
STAGE2_PROMPT = Path(
    "E:/type10-7/automation_reports/CV-SincNet/"
    "automation_prompt_backups/20260615_001820_stage2_closed_loop_v4/"
    "stage2_prompt.md"
)
FLOW_DOC = Path("E:/type10-7/automation_reports/CV-SincNet/stage2_automation_flow.md")
CONTRACT = Path("E:/type10-7/tools/optimizer_workflow_contract.md")
PROJECT_DOC = Path("E:/type10-7/项目.md")


class MonitorOptimizerClosedLoopPromptTest(unittest.TestCase):
    def test_prompt_removes_nonessential_launch_blockers(self):
        if not AUTOMATION_TOML.exists():
            self.skipTest(f"automation file not found: {AUTOMATION_TOML}")

        data = tomllib.loads(AUTOMATION_TOML.read_text(encoding="utf-8-sig"))
        prompt = data["prompt"]
        self.assertIn(data.get("status"), {"ACTIVE", "PAUSED"})

        required_tokens = [
            "N607 双 lane 闭环自动化薄包装",
            "按顺序读取并执行当前本地控制面",
            "不在 wrapper 中复制业务规则",
            "E:\\type10-7\\AGENTS.md",
            "E:\\type10-7\\项目.md",
            "E:\\type10-7\\tools\\optimizer_control_manifest.md",
            "stage2_prompt.md",
            "E:\\type10-7\\tools\\optimizer_workflow_contract.md",
            "stage2_optimizer_state.json",
            "USER_REQUIRED_SAFETY_STOP",
            "manifest、versioned prompt、contract、state 为准",
            "lane monitor 结果",
            "是否进入 optimizer/runner",
            "子 agent/多角色审查摘要",
            "SSH cleanup 状态",
        ]
        missing = [token for token in required_tokens if token not in prompt]
        self.assertEqual([], missing)

        forbidden_tokens = [
            "完整 post-run evidence audit",
            "闭环执行硬约束",
            "五角色",
            "四角色",
            "只有 lane-specific Evidence Analyst=PASS",
            "每个 completed lane 必须生成正好 8 个可执行实验",
            "激进分支必须新增 CVS 神经网络本体修改",
            "每批联邦 8 个候选中至少 2 个",
            "FedFishr 下一次 federated optimizer 强制落地验证",
            "十六个实验候选",
            "每张卡两个实验",
        ]
        present = [token for token in forbidden_tokens if token in prompt]
        self.assertEqual([], present)

    def test_versioned_prompt_and_flow_document_classify_gates(self):
        self.assertTrue(STAGE2_PROMPT.exists(), f"stage2 prompt not found: {STAGE2_PROMPT}")
        prompt = STAGE2_PROMPT.read_text(encoding="utf-8-sig")
        prompt_required = [
            "Rule Loading",
            "Operating Objective",
            "Idle launchable lane rule",
            "Monitor Module",
            "Optimizer Module",
            "Runner Module",
            "Forbidden Shortcuts",
            "Do not stop an idle lane",
        ]
        self.assertEqual([], [token for token in prompt_required if token not in prompt])

        self.assertTrue(FLOW_DOC.exists(), f"flow document not found: {FLOW_DOC}")
        flow = FLOW_DOC.read_text(encoding="utf-8-sig")
        flow_required = [
            "```mermaid",
            "Preflight",
            "Monitor",
            "Optimizer",
            "Runner",
            "Startup Health",
            "RouteEscalationGate",
            "ROUTE_MISMATCH_P0P1",
            "64-candidate matrix",
            "硬前进条件",
            "无所谓 gate 已移除",
            "仍保留的硬 gate",
        ]
        self.assertEqual([], [token for token in flow_required if token not in flow])

    def test_stage2_optimizer_prompt_is_cvs_specific_and_rich(self):
        self.assertTrue(STAGE2_PROMPT.exists(), f"stage2 prompt not found: {STAGE2_PROMPT}")
        prompt = STAGE2_PROMPT.read_text(encoding="utf-8-sig")
        required_tokens = [
            "Innovation And Rigor Module",
            "Evidence compression",
            "Divergent idea generation",
            "Idea cards",
            "Red-team pruning",
            "Matrix selection",
            "OA-MSE",
            "OPGAC",
            "OPGAC_NET",
            "JREF_C9_MULTICOMP_M2_E220",
            "stage2_priority_phase=OLD80_FIRST",
            "old_acc_target>=0.80",
            "opgac_metric_bundle",
            "opgac_score_table_required_columns",
            "MSE-lite",
            "MSE-Subspace",
            "OA-MSE-Head",
            "onboard_low_compute_training",
            "weibull_evt_required=true",
            "target_adapter_required=true",
            "pseudo_unknown_energy_required=true",
            "seen_new_evidence_gate_required=true",
            "seen_new_anchor_gate_required=true",
            "siamese_verifier_required=true",
            "accepted_only_online_update_required=true",
            "Phase2 Sample Protocol",
            "Stage2-A",
            "Stage2-B",
            "Stage2-C",
            "target/satellite receiver",
            "target_old_tx_ids=0,1,2,3,4,5",
            "satellite/LEO",
            "clean-view",
            "Subagent Review",
            "Validation Agent",
            "Runner Agent",
            "harm",
            "local code hook",
            "kill criterion",
            "failure signal",
            "GPU capacity plan",
            "route_family",
            "Subagent review summaries",
        ]
        self.assertEqual([], [token for token in required_tokens if token not in prompt])

    def test_contract_tracks_stage2_rich_optimizer_rules(self):
        self.assertTrue(CONTRACT.exists(), f"contract not found: {CONTRACT}")
        contract = CONTRACT.read_text(encoding="utf-8-sig")
        required_tokens = [
            "Current State Read Rule",
            "Gate Classes",
            "Idle Launchable Lane Obligation",
            "Monitor Boundary",
            "Lane Runner Contract",
            "Candidate Matrix Contract",
            "Phase2 Sample Protocol",
            "Metric Definitions",
            "Deployment View",
            "Evidence Sweep",
            "Route Retirement And Invalidity",
            "FULL_ARTIFACT_SWEEP_REQUIRED",
            "REMOTE_CURRENT_REQUIRED",
            "OA-MSE rows",
            "OPGAC rows",
            "route_family=OPGAC_NET",
            "JREF_C9_MULTICOMP_M2_E220",
            "opgac_metric_bundle",
            "old_unknown_hmean",
            "oa_mse_onboard_adaptation_bundle",
            "source_target_fusion_policy",
            "unknown_query_eval_only=true",
            "target_new_query_not_threshold_fit=true",
            "old_acc_target>=0.90",
            "seen_new_acc_target>=0.75",
            "target_channel_view=satellite/LEO",
            "Satellite/LEO target view is deployment-primary",
            "Clean view is a control/reference only",
            "Candidate-level retired-route gates",
            "tools is not a launch blocker by itself",
        ]
        self.assertEqual([], [token for token in required_tokens if token not in contract])

    def test_contract_prevents_global_phase2_local_patch_from_masking_launchable_rows(self):
        self.assertTrue(CONTRACT.exists(), f"contract not found: {CONTRACT}")
        contract = CONTRACT.read_text(encoding="utf-8-sig")
        required_tokens = [
            "PHASE2_LOCAL_PATCH_REQUIRED",
            "MUST NOT be set to `1` for the whole Phase2 lane when `launchability_summary.by_lane.phase2_spaceborne_fsl.runner_readiness=LANE_HAS_LAUNCHABLE_ROWS`",
            "local-patch deferral must be row-scoped",
            "launcher generation must omit or set `PHASE2_LOCAL_PATCH_REQUIRED=0`",
        ]
        self.assertEqual([], [token for token in required_tokens if token not in contract])

    def test_contract_requires_idle_launchable_lane_runner_obligation_and_gate_allowlist(self):
        self.assertTrue(CONTRACT.exists(), f"contract not found: {CONTRACT}")
        contract = CONTRACT.read_text(encoding="utf-8-sig")
        required_tokens = [
            "## Idle Launchable Lane Obligation",
            "current idle lane has `LANE_HAS_LAUNCHABLE_ROWS`",
            "must not return `MONITOR_ONLY_CONTINUE`",
            "The following are not launch blockers for that lane",
            "opposite lane active or awaiting completion audit",
            "older `latest_phase2_defer_result`",
            "subagent disagreement without a cited hard blocker",
            "metrics below target",
            "report-only, state-only, dry-run-only, or protocol-only",
            "hard blocker must be one of `Gate Classes`",
            "exact blocked outcome and artifact",
        ]
        self.assertEqual([], [token for token in required_tokens if token not in contract])

    def test_prompt_forwards_idle_launchable_lanes_to_runner_gate_allowlist(self):
        self.assertTrue(STAGE2_PROMPT.exists(), f"stage2 prompt not found: {STAGE2_PROMPT}")
        prompt = STAGE2_PROMPT.read_text(encoding="utf-8-sig")
        required_tokens = [
            "Idle launchable lane rule",
            "If validator `launchability_summary.by_lane` reports `LANE_HAS_LAUNCHABLE_ROWS`",
            "must enter Runner",
            "must not use `MONITOR_ONLY_CONTINUE`",
            "stale defer state, opposite-lane activity, metric under-target, or subagent disagreement",
            "contract's `Gate Classes`",
        ]
        self.assertEqual([], [token for token in required_tokens if token not in prompt])

    def test_project_doc_defines_multi_receiver_target_domain_sample_selection(self):
        self.assertTrue(PROJECT_DOC.exists(), f"project doc not found: {PROJECT_DOC}")
        project_doc = PROJECT_DOC.read_text(encoding="utf-8-sig")
        required_tokens = [
            "target receiver domain `R_t`",
            "`R_t` and `R_s` must be disjoint",
            "single `r_sat` is allowed but not mandatory",
            "must include target-old samples from `Y_old` and target-new samples from `Y_new`",
        ]
        self.assertEqual([], [token for token in required_tokens if token not in project_doc])

    def test_contract_and_prompt_require_target_domain_sample_coverage_before_launch(self):
        self.assertTrue(CONTRACT.exists(), f"contract not found: {CONTRACT}")
        self.assertTrue(STAGE2_PROMPT.exists(), f"stage2 prompt not found: {STAGE2_PROMPT}")
        contract = CONTRACT.read_text(encoding="utf-8-sig")
        prompt = STAGE2_PROMPT.read_text(encoding="utf-8-sig")
        required_tokens = [
            "target receiver domain may contain one or more receivers",
            "target receiver domain must be disjoint from CEN51 train receivers",
            "launchable Phase2 rows must expose target-old and target-new sample coverage",
            "do not require exactly one r_sat",
        ]
        self.assertEqual([], [token for token in required_tokens if token not in contract])
        self.assertEqual([], [token for token in required_tokens if token not in prompt])


if __name__ == "__main__":
    unittest.main()
