"""Patch or verify the standing N607 optimizer prompt."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import time
try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10 in ssr-gpu
    import tomli as tomllib
from datetime import datetime
from pathlib import Path


DEFAULT_AUTOMATION = Path(
    "E:/codex/home/automations/"
    "cv-sincnet-n607-monitor-optimizer-v4/automation.toml"
)
DEFAULT_REPORT_ROOT = Path("E:/type10-7/automation_reports/CV-SincNet")


FOUR_ROLE_REVIEW = """- 四角色子agent审查只提供风险提示，不作为 launch 前置硬门槛。四个角色固定为：
  1. 实验数据分析员：完整解析 completed lane 的 logs/metrics/config/checkpoint/stdout，给出 collapse、final-vs-best、SAT/final ranking、异常和可复现实验证据。
  2. 优化方向搜索员：基于数据证据搜索可落地优化方向，把方向转成候选参数、代码/脚本改动和 expected failure signals。
  3. 联网文献检索员：联网查询相关论文、方法和实现线索，记录来源、年份、方法要点以及与本代码 knob/小改动的映射；网络失败只记录原因，不阻断可验证实验启动。
  4. 监督审查员：审查不合理实现、无意义限制、近重复候选、AGENTS 风险、路径/registry/容量风险，并把问题转成修复建议或仅阻断对应危险候选。
- 四角色发现的问题必须转成候选参数、代码/脚本修复、验证命令或报告风险项；不得仅凭角色 verdict 停在 no-launch。"""


LITERATURE_AND_REVIEW = """- 文献/方法搜索只作为优化灵感和风险记录；搜索失败或没有新论文不阻断本地可验证候选启动。
""" + FOUR_ROLE_REVIEW


CANDIDATE_AND_LAUNCH = """候选与实验落地：
- 实验落地优先：optimizer 分析完成证据后，目标不是产出 blocked-only 设计矩阵，而是把可行候选落地为本地代码/脚本、验证、同步和实验启动。
- 显卡空闲时流程一定落实到实验任务启动：当 inventory 显示 gpu_compute=[]、active_training_processes=[]、launcher_context=[]、unknown_training_active=false，且存在 completed lane 时，必须选择可执行候选并推进到本地验证、SCP、远程 dry-run、容量复查和 launch。不能用“还要更多分析”“方向不够新”“缺 telemetry”“缺 readiness audit”作为最终状态。
 - 候选数量由证据、容量和可验证实现决定。当前 Stage2 闭环的默认是 64 个轻量 SFE/FTRC/telemetry 候选、每 GPU 8 个 slot、每 GPU 4-active 队列递补；非 Stage2 旧 lane 可按其合同数量执行。容量足够时尽量启动完整 queue，容量不足时启动所有安全可用 slot。不得因为某个激进候选暂不可实现而阻止其它可启动候选。
- 每个候选仍需记录 experiment_id、lane、parent_run/lineage、假设、对照、关键参数、GPU、run/log/report 路径、主要指标、风险、exact command 和 launchability_status；但这些字段是执行记录，不是额外阻断条件。
- missing telemetry、readiness audit、contract audit、prototype gate、collapse root cause 不能作为最终 no-launch 理由。它们必须被转成同轮可执行的代码/脚本/telemetry 修复、诊断候选或保守参数候选；只有修复或验证实际失败后，才记录对应 BLOCKED_*。
- unsupported CLI/code path 先修复或绕开：先检查现有 train.py、train_federated.py、train_cen31_distill.py、launcher、配置和测试表面；能用现有 knob/脚本实现就直接生成命令，不能实现就做最小本地补丁并验证。只有补丁超出安全小改、违反 AGENTS 或验证失败时，才允许 no-launch。
 - 可接受的 no-launch blocker 只限于：AGENTS 安全冲突、route/SSH 失败且桥接 fallback 也失败、local verification 失败、remote hash/语法/dry-run 失败、run/log 路径碰撞、当前容量策略上限、active/unknown training 风险、registry 证明同一 candidate/command/code 已 launched/running/completed。Stage2 轻量队列当前采用用户显式覆盖的每 GPU 4-active；重型 centralized/federated/FL 仍按 AGENTS 默认或另行显式授权。
- 维护 E:\\type10-7\\automation_reports\\CV-SincNet\\optimizer_execution_registry.jsonl。注册 key 至少包含 lane、evidence_hash 或 log_root/run_root、candidate_id、command_hash、code_hash、sync_mapping_hash、status、timestamp；不得重复 launch 已 LAUNCHED/RUNNING_CONFIRMED/STARTUP_HEALTH_PASS/COMPLETED_ANALYZED 的同一候选。"""


CENTRALIZED_STRATEGY = """集中式训练优化方向（当前策略）：
- CEN31/CEN31_C04 是当前集中式优先 parent/gate；除非最新证据反转，否则新集中式实验优先从它衍生。CEN_A31 只作为历史锚点或对照。
- 集中式候选优先使用已经支持并可本地验证的 knob、launcher 和小脚本改动落地：MixStyle/Fishr/GroupCE/DSQ 稳定性、SAT/信道增强、label smoothing、proto/SupCon、评估窗口、batch_best_joint、batch_best_risk_adjusted、selector/telemetry/reporting 修复。
- 网络本体修改、剪枝、结构搜索、物理前端拓扑和 TX/domain head 解耦可以作为激进候选，但不是启动前置条件。若本轮没有安全小改的结构实现，就先启动已有代码面可验证的集中式候选，并把结构方向记录为后续候选。"""


FEDERATED_STRATEGY = """联邦/VMB 训练优化方向（当前策略）：
 - federated/VMB 只保留 AGENTS 硬约束：WiSig train ratio=0.1，默认 epochs=200，默认 fl_rounds=200，默认 --fl_client_key receiver，每 GPU 默认最多两个训练进程（除非用户另行明确覆盖），非破坏性安全规则，以及本地优先验证/同步/启动流程。
- 不再添加莫名其妙的额外方向限制、参数禁令或固定配额。联邦/VMB 的参数、方向和候选数量由完成证据、失败模式、现有代码面和可验证修复决定。
- 若 FED/VMB 证据显示 R190-R196/R193 collapse/incomplete、SAT/final ranking 缺失、prototype/gradient/client-drift 诊断不足，自动化必须分析原因并转成代码/脚本/telemetry 修复候选；完成本地验证和远程 dry-run 后，在空闲 GPU 上启动诊断或修复实验。
- receiver-agnostic TX identification、VMB 原型/梯度机制、StyleBank、GRL/domain confusion、CEN31 经验迁移、SAT eval/ranking 都可以作为候选来源。除 AGENTS 硬约束外，不得因为某方向不符合预设模板而跳过启动。
- 诊断字段如 loss_rx_adv、loss_fishr、loss_dom、loss_cons、fed_proto_classes、prototype coverage、gradient cosine、client drift、global_eval_timing、round_train_time_s、round_eval_time_s 是优先观测项；缺失时优先补 telemetry 或脚本，而不是停工。"""


EXECUTOR_RULES = """Executor / launch 门禁：
- monitor_state=1 且存在 completed lane 时，进入 optimizer 后应继续推进实验落地；exact command、launcher、local verification、sync mapping、remote dry-run 和 startup health 是必须完成的工作步骤，不是可提前缺省的理由。
- 四角色子agent审查只提供风险提示，不作为 launch 前置硬门槛。角色审查发现的问题只能阻断对应危险候选，不能阻断其它已验证、容量允许、registry 不重复的候选。
 - 允许阻断 launch 的硬条件仅包括：AGENTS 安全冲突、route/SSH 失败且 fallback 不可用、local verification 失败、remote hash/语法/dry-run 失败、run/log 路径碰撞、当前容量策略上限、active/unknown training 风险、registry 重复、或用户明确要求暂停。
- AGENTS 落地顺序不可跳：先写 local report；只在本地 E:\\type10-7 或 E:\\type10-7\\code 修改；按需 snapshot；更新 SYNC_MANIFEST 或 report mapping；在 ssr-gpu 下做最小验证；先尝试 direct preflight；direct 不可行就尝试 AGENTS.md 已验证桥接路线；SCP 同步；远程语法/hash/dry-run 验证；短连接启动并记录 exact command/cwd/env/PID/GPU/log path/expected outputs/route_used；4-5 分钟内短连接复查 startup health；每次 SSH/SCP 后做本地 ssh.exe 到 N607/bridge 的残留检查。
 - 禁止删除/清理/kill/restart/relaunch 旧任务、remote-only 修改、未验证 relay/交互密码、包安装/服务变更、超过当前容量策略、覆盖既有输出、大范围未验证重构。"""


def load_toml(path: Path) -> dict:
    return tomllib.loads(path.read_text(encoding="utf-8-sig"))


def replace_key(text: str, key: str, value: object) -> str:
    rendered = json.dumps(value, ensure_ascii=False) if isinstance(value, str) else str(value)
    pattern = re.compile(rf"^{re.escape(key)}\s*=.*$", re.MULTILINE)
    if not pattern.search(text):
        raise ValueError(f"missing TOML key: {key}")
    return pattern.sub(lambda _match: f"{key} = {rendered}", text)


def replace_between(prompt: str, start: str, end: str, replacement: str) -> str:
    pattern = re.compile(rf"\n\n{re.escape(start)}.*?(?=\n\n{re.escape(end)})", re.DOTALL)
    if not pattern.search(prompt):
        raise ValueError(f"could not replace section {start!r} before {end!r}")
    return pattern.sub(lambda _match: "\n\n" + replacement, prompt, count=1)


def patch_prompt(prompt: str) -> str:
    if "第二阶段闭环自动化" in prompt and "stage2_optimizer_state.json" in prompt:
        return prompt
    prompt = re.sub(
        r"\n\n闭环执行硬约束：.*?(?=\n\n集中式训练优化方向)",
        "",
        prompt,
        flags=re.DOTALL,
    )
    prompt = re.sub(
        r"- 如果网络可用，做文献/方法搜索.*?Safety Reviewer 必须检查 AGENTS/SSH/GPU/同步/启动风险。",
        lambda _match: LITERATURE_AND_REVIEW,
        prompt,
        flags=re.DOTALL,
    )
    prompt = re.sub(
        r"- 文献/方法搜索只作为优化灵感和风险记录；搜索失败或没有新论文不阻断本地可验证候选启动。\n- 五角色审查只提供风险提示，不作为 launch 前置硬门槛。.*?不得仅凭角色 verdict 停在 no-launch。",
        lambda _match: LITERATURE_AND_REVIEW,
        prompt,
        flags=re.DOTALL,
    )
    prompt = prompt.replace(
        "五角色审查只提供风险提示，不作为 launch 前置硬门槛。",
        "四角色子agent审查只提供风险提示，不作为 launch 前置硬门槛。",
    )
    prompt = prompt.replace("五角色 verdict", "四角色子agent verdict")
    prompt = prompt.replace("五角色", "四角色子agent")
    if "候选与实验落地：" not in prompt:
        prompt = replace_between(prompt, "8 个候选矩阵：", "集中式训练优化方向", CANDIDATE_AND_LAUNCH)
    if "集中式训练优化方向（当前策略）：" not in prompt:
        prompt = replace_between(prompt, "集中式训练优化方向（当前最高策略）：", "联邦/VMB 训练优化方向", CENTRALIZED_STRATEGY)
    if "联邦/VMB 训练优化方向（当前策略）：" not in prompt:
        prompt = replace_between(prompt, "联邦/VMB 训练优化方向（当前最高策略）：", "Executor / launch 门禁", FEDERATED_STRATEGY)
    prompt = replace_between(prompt, "Executor / launch 门禁：", "最终中文报告必须说明", EXECUTOR_RULES)
    return prompt


def patch_automation(path: Path, report_root: Path) -> dict:
    raw = path.read_text(encoding="utf-8-sig")
    data = load_toml(path)
    old_prompt = data["prompt"]
    new_prompt = patch_prompt(old_prompt)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = report_root / "automation_prompt_backups" / f"{stamp}_unblock_launch_v4"
    backup_dir.mkdir(parents=True, exist_ok=True)
    workspace_backup = backup_dir / "automation.toml.before_unblock_launch"
    shutil.copy2(path, workspace_backup)

    external_backup = path.with_name(f"automation.toml.bak_unblock_launch_{stamp}")
    shutil.copy2(path, external_backup)

    updated = replace_key(raw, "prompt", new_prompt)
    updated = replace_key(updated, "updated_at", int(time.time() * 1000))
    path.write_text(updated, encoding="utf-8")

    final_data = load_toml(path)
    final_prompt = final_data["prompt"]
    required = [
        "实验落地优先",
        "显卡空闲时流程一定落实到实验任务启动",
        "四角色子agent审查只提供风险提示，不作为 launch 前置硬门槛",
        "实验数据分析员",
        "优化方向搜索员",
        "联网文献检索员",
        "监督审查员",
        "federated/VMB 只保留 AGENTS 硬约束",
    ]
    missing = [token for token in required if token not in final_prompt]
    if missing:
        raise RuntimeError(f"prompt update missing required tokens: {missing}")

    report = backup_dir / "update_summary.json"
    result = {
        "automation_toml": str(path),
        "workspace_backup": str(workspace_backup),
        "external_backup": str(external_backup),
        "old_prompt_chars": len(old_prompt),
        "new_prompt_chars": len(final_prompt),
        "prompt_changed": old_prompt != final_prompt,
        "updated_at": final_data.get("updated_at"),
    }
    report.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--automation", type=Path, default=DEFAULT_AUTOMATION)
    parser.add_argument("--report-root", type=Path, default=DEFAULT_REPORT_ROOT)
    args = parser.parse_args()

    result = patch_automation(args.automation, args.report_root)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
