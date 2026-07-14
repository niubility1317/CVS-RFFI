# Phase1 P0 Factorial8实验报告

## 基本信息

- 实验ID：`phase1_dgleo_p0factorial8_20260714`
- 日期：2026-07-14
- 执行者：Codex
- 阶段：Phase1 source-only地面训练，open-set仅作为内部P0代理诊断，不构成真实unknown拒识或Phase2/Phase3成功证据
- 目标：在不牺牲strict UDU、receiver floor和LEO弱信道性能的前提下，验证known正样本覆盖、TX条件域不变性和可微拒识风险三组机制对open-set代理几何的主效应及交互效应。
- 对比目标：`phase1_dgleo_corepath8_20260714`同seed候选及历史`ADV3B02_CORE90_SOFT_E200`。

## 协议与矩阵

- 数据：`ManySig.pkl`，`split_mode=tx_rx_day_1_7_2`，`labeled/unlabeled/source_val=0.08/0.72/0.20`，不使用目标接收机域。
- 训练：120 epochs，open-set相关损失从epoch 1启用，`concat_sa`全程启用。
- checkpoint：仅保留训练结束权重；source validation重评只用于训练健康诊断，不选择中间权重。
- 训练星地增强族：`clear_leo,low_elev_leo,rain_leo`。
- 正式测试增强族：`leo_clear_weak,leo_low_elev_weak,leo_rain_weak`；与训练族分离，避免只证明已知增强分布内鲁棒性。
- A：own-class invariant core正样本覆盖、positive-first、frozen reference bank、U_s三态quota与有效梯度闭环。
- B：TX条件receiver/day/channel不变性、receiver-aware local component、source episode density/overlap和星地局部一致性。
- C：对proxy/bridge/low-density/tail/overflow/radius-to-inter及query inter/overlap的可微直接风险优化。

| candidate | GPU | A | B | C | 主要用途 |
|---|---:|---:|---:|---:|---|
| P0F_A0B0C0 | 1 | 0 | 0 | 0 | 同配置基线 |
| P0F_A0B0C1 | 0 | 0 | 0 | 1 | 仅负风险，验证reject-all/无正覆盖失败模式 |
| P0F_A0B1C0 | 2 | 0 | 1 | 0 | 域不变性主效应 |
| P0F_A0B1C1 | 6 | 0 | 1 | 1 | 无positive-first时B+C交互 |
| P0F_A1B0C0 | 5 | 1 | 0 | 0 | 正样本覆盖主效应 |
| P0F_A1B0C1 | 4 | 1 | 0 | 1 | 正覆盖+直接风险 |
| P0F_A1B1C0 | 7 | 1 | 1 | 0 | 正覆盖+域不变性 |
| P0F_A1B1C1 | 3 | 1 | 1 | 1 | 全量联合机制 |

## P0资源门控修复

底层`dualguard16`门控不再要求GPU完全空闲。门控按`run_id/candidate_id/output_dir/process-group`归并为实验身份，不按CUDA子进程数量误计；当目标run未在该GPU运行、现有实验属于无关run、总实验数加入本次任务后不超过2且空闲显存不少于10000 MiB时，允许直接启动本次一个Phase1候选。未知PID身份、同run重复任务、第三个实验或显存不足均fail-closed。

## 本地修改与验证

| 文件 | 作用 |
|---|---|
| `code/scripts/launch_phase1_dgleo_dualguard16_20260712.py` | 实验身份级GPU容量门控，允许容量安全的无关实验共享 |
| `code/scripts/launch_phase1_dgleo_p0factorial8_20260714.py` | 8候选全因子矩阵及每卡一个候选 |
| `code/SSDG/train_ssdg.py` | positive-first、U_s三态路由、有效梯度与联合调度 |
| `code/cvsrffi/losses.py` | own-class known coverage及可微query风险损失 |

- Git分支：`codex/cvs-rffi-release-20260626`
- 关键提交：`29e415b`、`a304d60`、`40d5c33`
- 验证：`conda run -n ssr-gpu python -m pytest -q ...`，108项通过，4条PyTorch AMP弃用警告，无失败。
- 矩阵artifact：`local_artifacts/phase1_dgleo_p0factorial8_matrix_20260714.json`

## 成功标准与失败判据

联合推进要求同一候选同时满足：clean strict UDU、receiver floor和LEO weak mean/floor不低于对照容忍带；fixed `endpoint_accept_v1`下known hard TPR不塌缩；`source_episode_overflow`、`proxy_vaccept`、bridge/low-density/tail/overflow accept、`radius_to_inter_ratio`和`zid_p95/p99/tail_cvar`至少多数方向改善。动态`dm_*`改善而fixed endpoint或legacy proxy不改善，只能判为训练代理优化，不能promote。

以下任一情况判失败或阻断promotion：同run重复启动；未知GPU进程身份；每卡总实验数超过2；空闲显存低于10000 MiB；known hard TPR显著塌缩；p99相对最佳点扩张超过2.0度阻断final export、超过3.5度阻断promotion；U_s direct长期idle；strict UDU或最弱receiver出现不可接受回落。

## N607落地

- 本地代码承载：`E:\type10-7\github_publish\CVS-RFFI-repo`
- 远端根目录：`/home/szu2070436088/2510044040/CV-SincNet`
- Python环境：`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`
- 计划命令：`nohup <python> -u code/scripts/launch_phase1_dgleo_p0factorial8_20260714.py --allow-unrelated-compute --max-total-compute-per-gpu 2 --min-free-memory-mib 10000 ...`
- 计划日志：`logs/phase1_dgleo_p0factorial8_20260714/`
- 计划输出：`runs/phase1_dgleo_p0factorial8_20260714/`
- GPU分配：见矩阵表，每卡一个Phase1候选；允许同卡已有一个与本run无关且低显存占用的实验。
- 预计时长：7-9小时；10小时wall-clock为硬上限。

## 启动与完成状态

待同步、远端dry-run和启动后补充PID、实际命令、GPU快照、日志健康及最终逐候选结果。
