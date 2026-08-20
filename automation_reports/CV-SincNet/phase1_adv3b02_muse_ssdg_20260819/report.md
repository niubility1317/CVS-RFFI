# ADV3B02-MUSE-SSDG Phase1最小预登记

## 2026-08-20正式N607运行预登记

- 运行ID：`phase1_adv3b02_muse_ssdg_20260820_e5b321b`；输出根：`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3b02_muse_ssdg_20260820_e5b321b`，禁止覆盖。
- 固定实现：`work/cvs-active`合并提交`e5b321bdf95dc8a696a12b6d64f0fbc9405da603`；该提交的两个父提交为主线`ed890015ddb9968663609727c28fdb4d749d4334`与最终复审通过的MUSE提交`e767bad0082b3564f01c1d765b543a1780aa03d6`。
- 候选与资源：单seed`392002`、200epoch；GPU0顺序运行`M0,M1`，GPU1顺序运行`M2,M3`。同一主Agent为唯一launch owner，不并行重复启动任何候选。
- 数据与边界：`ManySig.pkl`；`tx_rx_day_1_7_2`；`L_s/U_s/V_cal/V_select=0.07/0.63/0.15/0.15`；训练期零目标接收机访问；checkpoint选择固定为`final_only`。
- 必需评测：每个完成训练的候选必须由同一最终checkpoint执行一次联合评测，并分别保留`clean`、`leo_clear_weak`、`leo_low_elev_weak`、`leo_rain_weak`日志与metrics。任一场景失败不得写`ARTIFACTS_COMPLETE`。
- N607只读预检：2026-08-20T09:54:06+08:00直连普通账户成功；项目根、`Dataset_WigSig/ManySig.pkl`和`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`可见；8块RTX3090均为1MiB/0%且无compute app或既有项目训练任务。
- 停止规则：仅在协议/路径/checkout/output冲突、训练或评测执行错误、缺失final checkpoint、缺失prediction/metrics闭合、OOM/NaN或相同确定性启动前异常时停止对应运行；不得因性能高低停止。
- 实际release提交：`29316416cd4fed806fe1030562c0204448f09681`；归档提交包含预登记报告，实际训练代码仍由`e5b321bdf95dc8a696a12b6d64f0fbc9405da603`固定。
- release映射：本地`E:/type10-7/local_artifacts/releases/phase1_adv3b02_muse_ssdg_20260820_e5b321b.tar.gz`→远端`/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_adv3b02_muse_ssdg_20260820_e5b321b.tar.gz`→解压根`/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_adv3b02_muse_ssdg_20260820_e5b321b`。
- 唯一release归档SHA-256：本地与远端均为`03f0585ec8184dab6a28d79ddebdbca07d021f8b4c0df6f2ee498995ee3bf505`；远端Python编译、launcher语法检查和M3 dry-run均通过，dry-run恰好产生1条训练命令与1条联合评测命令，且未创建run root。
- 实际启动命令框架：GPU0使用`--only=M0,M1`，GPU1使用`--only=M2,M3`；共同设置`ROOT=<release根>`、`RUN_ID=phase1_adv3b02_muse_ssdg_20260820_e5b321b`、`RUNS_ROOT=<正式输出根>`、`WISIG_PKL=<主项目ManySig.pkl>`和固定远端Python。

## 候选矩阵

| 候选 | 固定基座 | 能力 | seed | epoch | source角色比例 | checkpoint选择 |
|---|---|---|---:|---:|---|---|
| M0 | `ADV3B02_CORE90_SOFT_E200` | 同协议ADV3B02控制；不进入MUSE能力路径 | 392002 | 200 | `0.07/0.63/0.15/0.15` | `final_only` |
| M1 | 同M0 | 基础domain/GRL/self/nuisance | 392002 | 200 | 同M0 | `final_only` |
| M2 | 同M0 | M1+fusion+H/M/L路由 | 392002 | 200 | 同M0 | `final_only` |
| M3 | 同M0 | M2+satellite student+cross-receiver+classification prototype | 392002 | 200 | 同M0 | `final_only` |

四个候选固定同一`tx_rx_day_1_7_2`数据split及`L_s/U_s/V_cal/V_select`角色定义，均以`len(U_s loader)`作为每epoch optimizer step预算。M0只按该长度循环L_s，不读取U_s batch、不计算U_s损失、不创建MUSE state。四臂共同启用ADV3B02 PAIC guard：`enabled=true`、`sat_ce_delta=0.12`、`grad_delta=3.0`、`reliable_drop=0.01`、`cooldown_epochs=1`、`sat_scale=0.75`。

## Commit

- Tasks 1–7代码HEAD：`4c66489ea058f5fe8401c29a237a58708bd7451f`，固定本报告审计的3个生产文件实现。
- Task 8修复前文档提交：`66ba28c48f5961100483cf6794252e15ca9bfb3b`（`docs: close MUSE SSDG implementation evidence`）。
- Task 8 fix round 1文档提交：本文件所在修复提交；提交后的精确OID、push状态和远端分支读回记录在Git忽略的`.superpowers/sdd/2026-08-19-adv3b02-muse-ssdg/task-8-report.md`“Fix round 1”节。文档提交不改变上述代码HEAD。

## 命令

```bash
bash code/scripts/launch_phase1_adv3b02_muse_ssdg_20260819.sh --only=M0,M1,M2,M3
```

本次Task 7 fix round 1仅执行本地`bash -n`、pytest、`--dry-run --only=M3`及临时fake trainer/evaluator非dry-run控制流，不连接N607、不启动真实训练。

## 环境与CWD

- 计划环境：N607普通账户，`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`。
- 计划CWD：`/home/szu2070436088/2510044040/CV-SincNet`。
- 本地验证环境：`C:/Users/lh594/.conda/envs/ssr-gpu/python.exe`。
- 本地验证CWD：`E:/type10-7/github_publish/CVS-RFFI-repo/.worktrees/adv3b02-muse-ssdg`。

## 输入、输出与GPU

- 输入：`${ROOT}/Dataset_WigSig/ManySig.pkl`。
- 输出根：`${ROOT}/runs/phase1_adv3b02_muse_ssdg_20260819/{M0,M1,M2,M3}`；已存在的候选根禁止覆盖。
- GPU：默认`GPU=0`，所有子命令映射为`CUDA_VISIBLE_DEVICES=${GPU}`与进程内`cuda:0`。

## 停止规则

- 训练命令非零退出、`final_ssdg.pth`缺失或为空时停止当前候选并保留全部产物。
- 联合clean+三LEO评测命令失败或联合artifact为空时写`EVAL_FAILED_JOINT`并保留训练产物；拆分时任一场景缺失、计数非法、准确率与计数不一致或对应日志/metrics为空时停止，写`EVAL_FAILED_<SCENARIO>`。
- 不因中间或最终性能高低停止。

## 预期artifact

每个候选根必须包含非空`train.log`、`config.json`、`final_ssdg.pth`、`eval_clean.log`、`eval_leo_clear_weak.log`、`eval_leo_low_elev_weak.log`、`eval_leo_rain_weak.log`、`metrics_clean.json`、`metrics_leo_clear_weak.json`、`metrics_leo_low_elev_weak.json`、`metrics_leo_rain_weak.json`。真实评测器只调用一次并生成联合JSON；控制层从真实row计数重算四个场景aggregate，四份JSON顶层`scenario`、`aggregate.scenario`和row语义必须一致。仅当四组评测日志与metrics均非空时，`status.txt`才写`ARTIFACTS_COMPLETE`。

## Task 8：追踪闭合与发布准备

### 状态

Tasks 1–7实现已完成Task 8本地聚焦验证和正反追踪。当前结论是`LOCAL_IMPLEMENTATION_VERIFIED_WITH_RUNTIME_EVIDENCE_PENDING`，不是`ARTIFACTS_COMPLETE`或`ANALYZED`：MUSE-002实际loader receipt、MUSE-014真实M0–M3矩阵和MUSE-018训练外precision诊断入口尚未闭合；没有clean或三种LEO弱场景的性能结果。

### 完整聚焦测试

- 环境：`C:/Users/lh594/.conda/envs/ssr-gpu/python.exe`；解释器前缀读回为`C:/Users/lh594/.conda/envs/ssr-gpu`。
- CWD：`E:/type10-7/github_publish/CVS-RFFI-repo/.worktrees/adv3b02-muse-ssdg`。
- 文件映射：brief列出的12个测试文件均实际存在，无需使用等价文件替换。
- 命令：`python -m pytest code/tests/test_muse_ssdg_schedule.py code/tests/test_muse_ssdg_routing.py code/tests/test_muse_ssdg_losses.py code/tests/test_muse_ssdg_memory.py code/tests/test_muse_ssdg_training_heads.py code/tests/test_muse_ssdg_train_integration.py code/tests/test_muse_ssdg_satellite.py code/tests/test_muse_ssdg_checkpoint.py code/tests/test_phase1_muse_launcher.py code/tests/test_meta_ssl_pseudo_gate.py code/tests/test_concat_sat_channel_aug.py code/tests/test_phase1_p1_protocol.py -q`。
- 结果：退出码0；12个文件共收集107项，107项全部通过；测试进程未输出warning，也没有warning升级为error。

### 真实checkpoint无query smoke

- checkpoint：`E:/type10-7/automation_reports/CV-SincNet/qknnv42_strict_dual125_20260714_183556/artifacts/best_joint_safe_ssdg.pth`，8,582,116字节。
- checkpoint SHA-256：`2699eedcafe8cec880828592d2d65ba3781a9948939da5cf5c82b47143d59c98`。
- 路径发现方式：先在仓库报告中定向检索已登记的ADV3B02路径，再仅对3个精确候选路径执行只读`test -f`；未扫描数据根、未连接N607。
- 输入边界：CPU、1个batch、batch size 2；输入为确定性构造的source-shaped张量；`dataset_read_count=0`、`support_input_count=0`、`query_input_count=0`、`target_truth_read_count=0`。
- 执行结果：严格重建0 missing、0 unexpected；前向输出`tx_logits=[2,6]`、`z_id=[2,160]`且有限；反向、optimizer step、MUSE训练态保存和重新加载全部完成；退出码0。
- 输出artifact：`E:/type10-7/local_artifacts/adv3b02_muse_ssdg_task8_20260820/m3_real_checkpoint_no_query_smoke.pt`，274,118字节；第二个独立进程以`weights_only=True`回读schema、batch数、checkpoint严格加载标志和零query/truth计数，全部一致。
- 非失败关注：模型前向输出1条`torch.cuda.amp.autocast`弃用`FutureWarning`。该warning来自既有`code/model.py`调用，不影响本次退出码和数值有限性；完整聚焦pytest没有输出该warning。

实际承载方式：命令由`C:/Program Files/Git/bin/bash.exe`在上述CWD中执行，通过quoted here-doc把Python源码直接送入`ssr-gpu`解释器stdin；当时没有创建临时`.py`脚本。以下为实际运行的主smoke命令，未重构为新的入口：

```bash
mkdir -p /e/type10-7/local_artifacts/adv3b02_muse_ssdg_task8_20260820
PYTHONPATH=code PYTHONUTF8=1 PYTHONIOENCODING=utf-8 /c/Users/lh594/.conda/envs/ssr-gpu/python.exe - <<'PY'
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from post_stage_common import load_checkpoint
from cvsrffi.checkpoint_loading import build_exact_ssdg_model_from_checkpoint
from SSDG import train_ssdg

checkpoint_path = Path(r"E:/type10-7/automation_reports/CV-SincNet/qknnv42_strict_dual125_20260714_183556/artifacts/best_joint_safe_ssdg.pth")
artifact_path = Path(r"E:/type10-7/local_artifacts/adv3b02_muse_ssdg_task8_20260820/m3_real_checkpoint_no_query_smoke.pt")
device = torch.device("cpu")

checkpoint = load_checkpoint(str(checkpoint_path), device)
input_len = int((checkpoint.get("args") or {}).get("wisig_out_len", 256))
model, load_audit = build_exact_ssdg_model_from_checkpoint(
    checkpoint,
    input_len=input_len,
    device=device,
)
model.train()

args = train_ssdg.build_arg_parser().parse_args([
    "--output_dir", str(artifact_path.parent / "unused_output"),
    "--use_muse_ssdg", "true",
    "--muse_level", "M3",
    "--epochs", "200",
    "--checkpoint_selection", "final_only",
])
muse_state = train_ssdg._initialize_muse_training_state(args, model, device)
assert muse_state is not None and muse_state["level"] == "M3"
muse_state["schedule_state"] = train_ssdg.muse_schedule_for_epoch(69, muse_state["config"])

batch_size = 2
x = torch.linspace(-1.0, 1.0, steps=batch_size * 2 * input_len, device=device).reshape(batch_size, 2, input_len)
synthetic_source_labels = torch.tensor([0, 1], device=device, dtype=torch.long)
source_domains = torch.tensor([0, 1], device=device, dtype=torch.long)

optimizer = torch.optim.SGD(train_ssdg._optimizer_parameters(model, muse_state), lr=1e-5)
optimizer.zero_grad(set_to_none=True)
output = model(x, return_aux=True)
logits = output["tx_logits"]
z_id = output["z_id"]
z_dom = output["z_dom"]
heads = muse_state["heads"]
local_prob = heads.local_prob(z_id, source_domains)
loss = (
    F.cross_entropy(logits, synthetic_source_labels)
    - 0.05 * local_prob.clamp_min(1e-8).log().mean()
    + 0.05 * heads.self_supervised_loss(z_id, z_id * 0.99)
    + 0.05 * heads.nuisance_loss(
        z_dom,
        torch.zeros(batch_size, int(args.muse_nuisance_dim), device=device),
        torch.ones(batch_size, dtype=torch.bool, device=device),
    )
)
assert torch.isfinite(loss)
tracked = next(parameter for parameter in model.parameters() if parameter.requires_grad)
before = tracked.detach().clone()
loss.backward()
assert tracked.grad is not None and torch.isfinite(tracked.grad).all()
optimizer.step()
parameter_delta = float((tracked.detach() - before).abs().max().item())

state_payload = train_ssdg._muse_checkpoint_state(muse_state)
artifact = {
    "artifact_schema": "adv3b02_muse_ssdg_m3_no_query_smoke_v1",
    "checkpoint_path": str(checkpoint_path),
    "checkpoint_sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
    "checkpoint_load_audit": load_audit,
    "muse_level": "M3",
    "batch_count": 1,
    "batch_size": batch_size,
    "input_origin": "deterministic_source_shaped_tensor_no_dataset_read",
    "dataset_read_count": 0,
    "support_input_count": 0,
    "query_input_count": 0,
    "target_truth_read_count": 0,
    "forward_finite": bool(torch.isfinite(logits).all() and torch.isfinite(z_id).all() and torch.isfinite(z_dom).all()),
    "backward_finite": True,
    "optimizer_step_complete": True,
    "parameter_delta_max": parameter_delta,
    "tx_logits_shape": list(logits.shape),
    "z_id_shape": list(z_id.shape),
    "z_dom_shape": list(z_dom.shape),
    "loss": float(loss.detach().item()),
    **state_payload,
}
torch.save(artifact, artifact_path)

restored_artifact = torch.load(artifact_path, map_location="cpu", weights_only=True)
restored_state = train_ssdg._initialize_muse_training_state(args, model, device)
train_ssdg._restore_muse_checkpoint_state(restored_state, restored_artifact)
for key, value in muse_state["heads"].training_state_dict().items():
    assert torch.equal(restored_state["heads"].training_state_dict()[key], value)
assert restored_state["schedule_state"] == muse_state["schedule_state"]
assert restored_artifact["batch_count"] == 1
assert restored_artifact["query_input_count"] == 0
assert restored_artifact["target_truth_read_count"] == 0
assert restored_artifact["forward_finite"]
assert restored_artifact["optimizer_step_complete"]

summary = {
    "status": "VERIFIED",
    "artifact": str(artifact_path),
    "artifact_bytes": artifact_path.stat().st_size,
    "checkpoint": str(checkpoint_path),
    "checkpoint_sha256": artifact["checkpoint_sha256"],
    "checkpoint_load_strict": load_audit["checkpoint_load_strict"],
    "missing_keys": load_audit["missing_keys"],
    "unexpected_keys": load_audit["unexpected_keys"],
    "batch_count": 1,
    "query_input_count": 0,
    "target_truth_read_count": 0,
    "forward_shape": list(logits.shape),
    "z_id_shape": list(z_id.shape),
    "loss_finite": True,
    "optimizer_step_complete": True,
    "muse_state_roundtrip": True,
}
print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
PY
```

主smoke已记录输出：

```text
E:\type10-7\github_publish\CVS-RFFI-repo\.worktrees\adv3b02-muse-ssdg\code\model.py:695: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
  with torch.cuda.amp.autocast(enabled=False):
{"artifact": "E:\\type10-7\\local_artifacts\\adv3b02_muse_ssdg_task8_20260820\\m3_real_checkpoint_no_query_smoke.pt", "artifact_bytes": 274118, "batch_count": 1, "checkpoint": "E:\\type10-7\\automation_reports\\CV-SincNet\\qknnv42_strict_dual125_20260714_183556\\artifacts\\best_joint_safe_ssdg.pth", "checkpoint_load_strict": true, "checkpoint_sha256": "2699eedcafe8cec880828592d2d65ba3781a9948939da5cf5c82b47143d59c98", "forward_shape": [2, 6], "loss_finite": true, "missing_keys": 0, "muse_state_roundtrip": true, "optimizer_step_complete": true, "query_input_count": 0, "status": "VERIFIED", "target_truth_read_count": 0, "unexpected_keys": 0, "z_id_shape": [2, 160]}
```

同一次原始Git Bash调用随后执行以下独立readback命令；它重新打开落盘artifact，不复用主进程内存：

```bash
PYTHONPATH=code PYTHONUTF8=1 PYTHONIOENCODING=utf-8 /c/Users/lh594/.conda/envs/ssr-gpu/python.exe - <<'PY'
from pathlib import Path
import torch
p=Path(r"E:/type10-7/local_artifacts/adv3b02_muse_ssdg_task8_20260820/m3_real_checkpoint_no_query_smoke.pt")
a=torch.load(p,map_location="cpu",weights_only=True)
assert a["artifact_schema"] == "adv3b02_muse_ssdg_m3_no_query_smoke_v1"
assert a["batch_count"] == 1 and a["query_input_count"] == 0 and a["target_truth_read_count"] == 0
assert a["checkpoint_load_audit"]["checkpoint_load_strict"] is True
print(f"ARTIFACT_READBACK_OK path={p} bytes={p.stat().st_size} schema={a['artifact_schema']}")
PY
```

readback已记录输出：

```text
ARTIFACT_READBACK_OK path=E:\type10-7\local_artifacts\adv3b02_muse_ssdg_task8_20260820\m3_real_checkpoint_no_query_smoke.pt bytes=274118 schema=adv3b02_muse_ssdg_m3_no_query_smoke_v1
```

该smoke只验证真实历史ADV3B02 checkpoint与M3训练路径、optimizer和MUSE state回环兼容，不使用真实source batch，也不产生准确率、DG收益、LEO鲁棒性或晋级证据。

### 18项正向追踪与反向审计

- 逐项状态：MUSE-001、003至013及017为`verified`；MUSE-002、014、015、016、018为`implemented`；`pending=0`。
- MUSE-002没有实际loader receipt证明四角色物理ID互斥、source/target receiver不相交和零target进入；synthetic smoke不提供该证据。
- MUSE-014只完成launcher dry-run、fake控制流和step预算测试；真实M0–M3单seed矩阵未运行，须以四臂真实run artifact升级状态。
- MUSE-018训练内precision遥测保持`N/A`，避免训练读取`U_s`真值；当前没有独立训练外precision诊断入口，因此该子要求与真实telemetry/泄漏探针结果均未闭合。
- 汇总：总要求18，`verified=13`、`implemented=5`、`pending=0`；实现映射18/18，但运行期证据未闭合。
- 生产文件审计范围：`0e1019beb8f9c3217b4ae84f1a56a4be6dd5ba9e..4c66489ea058f5fe8401c29a237a58708bd7451f`。
- 反向结果：`code/cvsrffi/muse_ssdg.py`映射MUSE-003至012、017；`code/SSDG/train_ssdg.py`映射MUSE-001至013、017、018；launcher映射MUSE-014至016。3/3个新增或修改生产文件均有规范来源，未发现需要删除或重新审批的规范外生产逻辑。
- 完整逐项证据见`analysis/adv3b02_muse_ssdg_traceability_20260819.md`。

### 单一release归档准备

拟定归档名：`adv3b02_muse_ssdg_code_4c66489ea058.tar.gz`。归档仅包含Tasks 1–7在代码身份`4c66489ea058f5fe8401c29a237a58708bd7451f`下的3个生产文件：

1. `code/cvsrffi/muse_ssdg.py`
2. `code/SSDG/train_ssdg.py`
3. `code/scripts/launch_phase1_adv3b02_muse_ssdg_20260819.sh`

Task 8不创建归档、不执行SSH/SCP、不启动实验。后续唯一runner在N607 preflight通过后创建这一个归档，只做一次本地/远端归档SHA比较，并在远端执行两个Python文件的`py_compile`和launcher的`bash -n`；不增加成员SHA、seal、receipt或额外发布gate。

### Task 8关注与下一状态

- 最高风险剩余项是尚无真实M0–M3单seed训练及其clean/三LEO逐场景结果；同时缺少MUSE-002实际loader receipt和MUSE-018独立训练外precision诊断入口。不能据当前本地证据判断性能、晋级或发表价值。
- MUSE-002须在实际loader receipt读回四角色物理ID互斥、source/target receiver不相交和target计数0后升级；MUSE-014须在真实四臂矩阵落盘后升级；MUSE-015、016、018须在真实训练、评测与训练外诊断闭合后升级。
- release归档创建、N607资源/路径preflight、单次SHA比对、远端编译、启动后PID/CWD/cmdline/GPU/log增长检查均留给后续唯一runner；本Task未越权执行。

## Final fix wave（FFR-1至FFR-7）

### 修复结论

- FFR-1：M2/M3第三路融合证据已改为`MUSEClassificationPrototypeBank`基于`z_id`产生的真实classification prototype概率；缺失类概率为0，概率有限且归一化。`L_s`标签与domain计数产生的全局/source-domain prior已在真实融合主链routing前执行alignment；不读取`U_s` TX truth。
- FFR-2：`proto_momentum`已控制有标签和稳定高置信未标注classification prototype的EMA更新，并与0.05–0.10未标注贡献分离。epoch 181进入S3C后，temporal memory、classification prototype、threshold/statistics、`L_s` prior和local teacher均冻结；后续`train()`与optimizer step不能改变local teacher。
- FFR-3：M3已按稳定SHA mask逐样本从strong或satellite/nuisance输出中唯一选择identity logits与`z_id`，H/M/L及相关identity consistency只消费所选分支；M1/M2强制只使用strong identity。
- FFR-4：MUSE训练入口在`--muse_external_final_eval true`时返回`DELEGATED_TO_MUSE_LAUNCHER`，不执行内部target held-out评测，也不生成`frozen_phase1_heldout_eval.json`；launcher保持唯一一次canonical joint target eval。非MUSE默认内部评测行为不变。
- FFR-5：formal evaluator新增strict checkpoint reconstruction模式，使用`strict=True`、禁止direct-builder fallback，并在任何missing、unexpected、shape mismatch或重建异常时于metrics写入前非零退出。launcher强制strict模式并验证`reconstruction_audit`，不合格时写`EVAL_FAILED_JOINT`而非`ARTIFACTS_COMPLETE`。未请求strict时保留旧fallback行为。
- FFR-6：`full_ablation_spec.py`和`phase1_ablation_factory.py`的活动生产配置已从parser非法的`source_validation_only`迁移为`final_only`；共享入口解析与formal final checkpoint角色测试已闭合。
- FFR-7：traceability与本报告已引用真实调用链测试；完整RED/GREEN、文件、commit及push/OID记录写入`.superpowers/sdd/2026-08-19-adv3b02-muse-ssdg/final-fix-report.md`。

### 真实调用链证据

- prototype与prior：`test_classification_prototype_probabilities_are_normalized_with_explicit_missing_classes`、`test_m2_fusion_uses_global_local_prototype_and_l_s_prior_alignment`。
- schedule与S3C冻结：`test_proto_momentum_boundary_is_095_then_099_at_s3b_and_s3c`、`test_prototype_momentum_and_unlabeled_contribution_are_distinct_controls`、`test_epoch_181_freezes_muse_statistics_prior_and_local_teacher_state`、`test_s3c_checkpoint_round_trip_restores_frozen_local_teacher_and_prior_state`。
- identity选择：`test_m3_sha_mask_selects_exactly_one_identity_student_per_row`、`test_m1_m2_never_enable_satellite_identity_student`。
- 唯一评测与strict恢复：`test_muse_can_delegate_final_target_eval_without_changing_legacy`、`test_fake_joint_evaluator_runs_once_and_writes_four_semantic_metrics_before_complete`、`test_strict_reconstruction_failure_exits_before_metrics_are_written`、`test_launcher_rejects_non_strict_or_fallback_reconstruction_metadata`。
- factory迁移：`test_active_phase1_row_factories_emit_parser_valid_final_only_selection`、`test_active_ablation_configs_pass_shared_checkpoint_parser`。

### 发布与证据边界

- final fix实现提交：`3f3809b1527c840a72f6ff75edd92c74cd87e085`（`fix: close MUSE SSDG final review findings`）；该提交已由post-commit hook推送并读回同OID，最终证据提交仍将在本轮结束时再次独立核对远端分支OID。
- 验证范围：聚焦RED/GREEN、完整MUSE/launcher/evaluator/protocol/factory pytest、changed Python `py_compile`、launcher `bash -n`、M3 dry-run、真实ADV3B02 one-batch no-query smoke和`git diff --check`。不连接N607，不执行target评测。
- final fix合并测试：16个文件、175项全部通过；退出码0。
- final fix真实checkpoint smoke：`E:/type10-7/local_artifacts/adv3b02_muse_ssdg_final_fix_20260820/m3_true_prototype_identity_strict_state_no_query_smoke.pt`，279,773字节；严格加载0 missing/0 unexpected；真实三头为global/local/prototype，prior alignment最大变化`0.06568282842636108`；稳定SHA mask选择2条strong与2条satellite；S3C strict状态回环和独立artifact回读通过；query、target truth与target eval计数均为0。
- 真实M0–M3训练：未运行。
- 真实clean及三LEO场景性能：未产生。
- 当前状态仍是本地实现与验证闭合，不是`ARTIFACTS_COMPLETE`、`ANALYZED`或性能晋级证据。

### Final fix后续单一release清单

旧Task 8的3成员归档计划已由本final fix覆盖。后续若由唯一runner执行N607发布，归档代码身份必须为`3f3809b1527c840a72f6ff75edd92c74cd87e085`，并包含`code/cvsrffi/muse_ssdg.py`、`code/SSDG/train_ssdg.py`、`code/scripts/launch_phase1_adv3b02_muse_ssdg_20260819.sh`和`code/scripts/eval_ssdg_sat_per_rx.py`共4个运行文件；strict evaluator不能遗漏。两个活动factory迁移文件不被MUSE运行链消费，不加入该归档。不在本final fix wave创建归档或连接N607。
