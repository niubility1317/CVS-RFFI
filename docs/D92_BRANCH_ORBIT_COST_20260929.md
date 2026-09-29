# BranchOrbitCE 资源口径核对

日期：2026-09-29。本文区分固定模型包、接收端实验缓存、单任务持久状态和前向计算次数。没有读取 probe 指标或 query 评分，没有加载真实 support/query，也没有重新执行实验。当前设计已有成本公式与 C26K20 数字，但未完整列出四档 K 和实际导出缓存口径，因此补充本文。

## 模型与通信

本地证据为 [readback_1790674538.json](E:/type10-7/automation_reports/CV-SincNet/20260929-phase2-d92-branch-orbit-ce-support-m4-r01/evidence/readback_1790674538.json) 中 `rx3-cvs-daot-rc4-s2026092701/export.log` 的 `BRANCH_ORBIT_SUPPORT_FEATURES_COMPLETE`。只检查导出完成记录，未检查同一文件中的 probe 日志。该记录绑定 capsule `residual-noeq-76121e6f34363fa612ec25fb`、模型 seed `2026092701`、checkpoint SHA256 `f7ea5064d56711c3173b16636a52af27018ba11a459f0205de950c134362e53b`。

| 项目 | 已核实数值 | 范围 |
|---|---:|---|
| 固定 Phase1 模型文件 | 15,992,872 B | 完整训练 checkpoint 包；不是最小推理包 |
| 新增源样本载荷 | 0 B | 本方法未使用源样本或源样本特征库 |
| 新增地面统计载荷 | 0 B | 本方法没有新增地面摘要或统计包 |
| 模型是否已部署、模型本轮增量传输量 | unknown / null | 现有证据无法判定部署位置及是否需要首次传输 |

因此不能概括为“全部通信为 0”。若需首次部署，模型包可能构成传输；接收数据、注册信息、结果或日志是否跨设备传输也取决于部署方案，本文没有测量这些链路。后面的缓存和状态字节是存储口径，不能自动当作地面到卫星的通信量。

## 完整实验缓存与单任务输入

上述单个模型、三 RX cohort 的导出包含 **8,273 条去重物理 support、900 个注册 split**，属于该 cohort 的实验全集。每条物理记录保存四个原生分支，各为 `[4,160]` float32，以及原 received IQ 的一份 `[96]` float32 FFT；派生 view 不增加物理样本数。每条特征数组为 `4 × (4 × 160 + 96) = 10,624 B`。

| 接收端产物 | 字节数 |
|---|---:|
| 五块特征数组 | 87,892,352 B |
| 原物理索引数组 | 66,184 B |
| 物理 ID 与类名数组 | 959,668 B |
| `support_branch_orbit_features.npz` 整文件 | **88,923,468 B** |
| 整文件加 startup、support_splits、provenance 元数据 | 96,093,375 B |

NPZ 整文件还含容器和绑定字段开销；最后一行不含完成 marker。上述数值不是单个 C=26、K=20 任务的内存，也不是八个模型/cohort lane 的总量。

单任务注册 C=26 类时，K 表示每类物理 support 数，总数 `N=26K`。下表“输入特征数组”仅计该任务所需的五块 float32 特征，不含缓存容器、ID、索引或日志。

| 每类 K | 物理 support N | 输入特征数组 | 候选持久数值状态 | support 原生前向次数 |
|---:|---:|---:|---:|---:|
| 1 | 26 | 276,224 B | 618,608 B | 104 |
| 5 | 130 | 1,381,120 B | 3,092,144 B | 520 |
| 10 | 260 | 2,762,240 B | 6,184,064 B | 1,040 |
| 20 | 520 | 5,524,480 B | 12,367,904 B | 2,080 |

这些是单次全 support `fit_branch_orbit_ce(..., arm='orbit_ce')` 的部署状态。当前 support 诊断的真正 K1 行仅做数值检查，没有拟合诊断头；表中 K1 状态由独立纯合成公共 fit 核算，不声称真实 K1 诊断已训练或获得独立持出结果。

## thin QR 修复后的状态核算

依据当前 [核心实现](../code/cvsrffi/d92_branch_orbit_ce.py) 的 `BranchOrbitCEState`（第 242 行起）和 `fit_branch_orbit_ce`（第 289 行起），orbit 状态持有如下七个 float64 数组和两个 float64 标量：

| 状态 | 数组形状 | 字节数 |
|---|---|---:|
| `support_background` | `[N,4,256]` | `8 × N × 4 × 256` |
| `support_auxiliary` | `[N,4,480]` | `8 × N × 4 × 480` |
| `alpha` | `[N,C]` | `8 × N × C` |
| `reference_kernel`、`center_mean`、`support_scale` | 各 `[N]` | 合计 `24 × N` |
| `target_mean` | `[C]` | `8 × C` |
| `reference_self`、`center_grand` | 两个标量 | 合计 16 |

总计 `8 × [N × (4 × 736 + C + 3) + C + 2] B`。C=26 时为 `23,784 × N + 224 B`。thin QR 是 `_orbit_parts` 的等价数值分解；当前实现没有将 Q/T 注册为额外持久字段，也没有因此删除保存的四 view support 特征。不能把 QR 临时数组重复计入持久状态，也不能据此宣称状态已被压缩。

`head_bytes` 和 `persistent_state_bytes` 在本实现中包含上述 support 特征与核系数，不能理解为只有一个小型线性分类矩阵。该口径排除类名、physical ID、audit/逐步日志、Python 对象、固定骨干、原缓存、Gram 矩阵与优化器临时工作区，不等于进程峰值内存或最小序列化部署包。

进行了一次有界合成核算：Python 为 `C:/Users/lh594/.conda/envs/ssr-gpu/python.exe`，NumPy 2.2.6，BLAS 限 2 线程；各 K 重新使用 PCG64 seed 930 生成独立标准正态 float32 五块输入，C=26，调用公共 `fit_branch_orbit_ce`，未调用 score/predict。逐一读取七个实际数组的 shape/dtype/nbytes，加上两个标量，再与 `state.audit_dict()` 的两个 bytes 字段及解析公式比较，四档全部精确相等。核算时核心文件 SHA256 为 `7633d65af7c2b84d66b06e833a5245e1b6580109b25d8b3fe56becfa9c89d287`。

四次合成 fit 的状态均为 `CONVERGED`。此处只用它们核对实际状态布局，不作耗时基准、物理性能判断或配置选择，没有扩大运行单元测试。

## 四 view 的计算口径

[导出器](../tools/export_d92_branch_orbit_support_features.py) 对每个物理 support 先读取一次 IQ，再分别前向 0°、90°、180°、270° 四个 view，每次 native batch size 为 1。因此 N 条物理 support 对应 **4N 次前向，不是 4K 次**。C=26 时为 `104K` 次。FFT 从原 received IQ 计算一次，不随 view 复制计算。

实际完成记录给出 `N=8273`、support 前向 `33092=4N` 次；另有 4 次纯合成 smoke 前向，总计 33,096 次。日志中的 `native_physical_forward_count` 在这个 exporter 中实际按 view 调用累计，唯一物理数应读取 `support_physical_observation_count=8273`，避免误读字段名称。

若后来正式使用该四 view 方法对一个新 query 推理，同样需要 4 次冻结原生前向和一份原始 FFT，另加接收端特征变换、核计算及对全部注册类的评分；不是只需 4 次前向便完成所有计算。当前纯数学 `state.score` 接收的是已提取特征，不会自行运行骨干。新 query 不更新 support 状态。

该导出 lane 的实测 native 前向时间为 176.922 秒、导出总时间 183.244 秒；进程峰值 RSS 为 1,275,387,904 B，包含导出运行环境等开销。它不是 GPU 显存、单任务持久状态或部署内存需求。一个导出完成 marker 只证明该缓存完成，不证明 probe 完成，更不提供 query 性能结论。
