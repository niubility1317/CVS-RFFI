# CVS 无增强轻量架构与常见网络基准

run_id：`20261001-phase1-cvs-clean-architecture-manysig-m32-r01`。当前状态：RUNNING（独立读回 VERIFIED）；源训练已启动，无本批 clean 测试结果。

## 目的与公平对照

用户要求纯交叉熵、不使用星地信道增强、只测 clean，并以无增强的常见网络作对照。本批固定 4 个基线：原 CVS 身份骨干、CVCNN、普通 1D CNN、轻量 1D ResNet。另有 4 个 CVS 候选：PA 正交化、矩池化、二者结合、共享复数骨干。共 8 架构 × 4 固定 seed = 32 行。

所有网络从零训练，保持同一物理划分。6300 条源域 L 参与训练，56700 条 U 不使用，27000 条源域 V 仅评估。每行 200 epoch × 50 次优化，batch 128、drop_last=False、AdamW lr=2e-4、wd=1e-4、cosine 最低 lr=1e-6。FP32、无梯度裁剪、无信道或数据增强、无域骨干、无 PL/teacher/EMA、无额外损失。相同 seed 使用独立且相同的 batch-loader 随机序列。

保留各基线的分类头差异：原 CVS 使用 scale=30 无 margin 的余弦分类头；常见网络使用 Linear 头。1D ResNet 是本项目六个 basic block 的轻量实现，不声称是作者原始 ResNet18。原 CVS 的网络内部 dropout 按原模型保留，与输入增强分开记录。

## 参数与局部计算测量

| 架构 | 参数数 | 每样本卷积及矩阵 MAC |
|---|---:|---:|
| 原 CVS | 382146 | 9862436 |
| PA 正交化 | 382146 | 9862436 |
| 矩池化 | 382338 | 9862436 |
| 正交 PA＋矩池化 | 382338 | 9862436 |
| 共享复数骨干 | 168682 | 3119232 |
| CVCNN | 104646 | 11797248 |
| 普通 1D CNN | 174726 | 11797248 |
| 轻量 1D ResNet | 167814 | 7955008 |

这些 MAC 不包含 FFT、归一化、物理基函数、池化及逐元素算子。全部 CPU 合成 benchmark 已完成，仅证明测量路径可执行；共享复数骨干 CPU 训练时间高于原 CVS，不能据参数或 MAC 宣称训练更快。N607 实际训练时间、推理时间、峰值显存、常驻状态和有效梯度参数将在每行训练后记录。benchmark 更新模型副本，不更新正式权重。

## 选择与测试权限

每行固定使用 E200 权重。仅用四 seed 最终源域 V 的平均准确率和最差源 RX 准确率，按 0.5/0.5 加权选择一个候选；0.2 个百分点以内优先较小 MAC，然后较少参数。四个基线始终保留。候选参数上限为原 CVS 的 1.1 倍。

源矩阵全部完成后冻结 `source_selection.json`，才为四个基线和唯一入选 CVS 候选进行 clean 预测及独立 truth-last 评分，共 20 行。源 release 不构建目标数据、不预测或评分目标。历史目标 benchmark 已经暴露，结果属于固定研究对照，不能当盲测；不据目标成绩调结构、重排候选或选择性重训。

## 验证与执行

14 个聚焦测试 PASS：8 架构真实 CE 前后向及 AdamW 更新、PA 正交与相位性质、池化、样本独立性、禁用增强/继承/目标的负测、源选择、全新进程导入及 profile 副本隔离。独立 P0/P1 审查 PASS，未发现阻断问题。本地 Torch 为 2.10.0+cu128；不据此声称 N607 Torch 2.1 已运行通过。32 行登记 `--launch-ready` 校验 VALID。

唯一 launch owner：`codex/root/cvs-clean-architecture-20261001`。远端每 GPU 最多两个总训练任务，并在启动前要求至少 12 GB 空闲显存。逐行输出独占；技术失败保留产物且不自动重试，不因低成绩停机，不干预其他 owner。

数据、全部六类 seed、逐行配置和产物位置见 [experiment.json](experiment.json)。完整设计见 [设计说明](../../../docs/CVS_CLEAN_ARCHITECTURE_DESIGN_20261001.md)。本地证据见 [独立审查](evidence/independent_review.json) 和 [CPU 测量](evidence/local_cpu_resource_profiles.json)。

## 当前交接

已完成实现、固定矩阵、登记、本地验证和独立审查。下一步提交并验证远端 Git，再发布独立 source-only release，读回真实进程、有效配置和逐轮日志。已有产物或进程时先核实，不重复启动。当前尚不能判断改进是否提高 clean 泛化。

## N607 发布与运行证据

代码 commit：`583be084d91dc1fd8540bbf1da33b61b91b38975`，已 push 并独立核实远端分支 OID 相同。不可变 release 解压、传输 SHA、远端 compile 和冷入口检查通过。独立读取 `/proc` 核实 dispatcher PID `700444`、CWD 和 argv；不是仅据提交回执判断运行。

本次读回：{'RUNNING': 16, 'QUEUED': 16}。已核实有效配置的行：native-s2026092701, orthogonal_pa-s2026092701, moment_pool-s2026092701, orthogonal_moment-s2026092701, shared_complex-s2026092701, cvcnn-s2026092701, real_cnn-s2026092701, resnet1d-s2026092701, native-s2026092702, orthogonal_pa-s2026092702, moment_pool-s2026092702, orthogonal_moment-s2026092702, shared_complex-s2026092702, cvcnn-s2026092702, real_cnn-s2026092702, resnet1d-s2026092702；实际无增强、无域骨干、无额外损失、target_access=false，6300/56700/27000 源角色计数和每轮 50 步吻合。硬件 RTX 3090，Torch 2.1.0+cu121。当前逐轮进度与 PID/GPU 见 [独立读回](evidence/source_launch_readback.json)。

源训练继续按既有队列执行。下一步只读监控所属 run，完成后核实 E200 checkpoint、完整日志、资源测量与源选择；随后为冻结的五个网络做 clean 测试。尚无新测试准确率，不能宣称任何架构提升。
