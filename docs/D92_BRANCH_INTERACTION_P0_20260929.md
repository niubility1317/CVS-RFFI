# D92-BranchInteraction support-only 诊断 P0/P1 审查

日期：2026-09-29。范围为本次已就绪核心、入口、prepare/run/publish/preflight、冻结配置与 support 汇总接入。只做下一次运行的直接正确性审查，不提出性能设计或超参数建议，不复审旧方法效果。

## 当前结论

最终结论：未发现尚未解决的 P0/P1 阻断。唯一发现的 P1 是汇总初版只检查跨 K 的 H、新类平均值，与预登记逐 K 要求不符；该处已在任何新结果产生前修复并完成定点读回。运行时输入、物理折隔离、中心化、配置绑定及发布路径未发现 P0/P1。本结论只覆盖本次 support-only 诊断，不代表真实运行已完成或 query 泛化已验证。

## 检查通过的边界

1. **只读 support 输入。** 新入口通过既有 load_support 读取 capsule manifest、明确绑定的 support cache 四份元数据和 support_branch_features.npz。该 reader 的精确成员白名单不接受 IQ、query 或旧适应状态；完整 Cartesian matrix、物理 ID/原索引/标签、六旧类前缀、每类 K、缓存 union 均核对。入口按当前任务 positions 切片后调用 core，没有调用 encoder、source loader、query scorer 或旧 probe 的拟合函数。导入 exporter 仅取得常量，不调用其导出功能；本地纯导入核验无 torch。
2. **checkpoint 与缓存来源。** load_support 核对真实 SHA、model seed、capsule/schema、source-only scratch provenance、epoch 200、空继承链及缓存各元数据一致性。新诊断不加载 checkpoint，也不继承头或归一化拟合状态。现有 producer 的 provenance 元数据不是对所有历史运行的系统调用级安全证明；本次不重复既有远端数据核验。
3. **每折重新估计。** core 按物理 ID 排序后逐类取模生成 F=min(K,3) 折。每折每臂仅用 keep 子集重新计算 Gram、reference、均值和 alpha；held 仅传给 score，三臂共用同一物理划分。特征块的 normalization 是单条样本函数，没有跨样本拟合。K1 在任何 _fit 前返回，OOF=null、0 分解，不能制造独立验证证据。
4. **核与中心化一致。** 三臂固定为 KB+KA、1.5(KB+KA)、KB+KA+KB*KA；ridge=1，对物理样本损失求和，截距不正则化。训练 reference 差分再中心化与 HGH 等价；held 行使用相同训练 reference/均值，单行相对固定 support 的核均值不读取其他 held 行。所有允许的折均保持每类相同 train K，故 centered one-hot 的数学均值为零。精确恒定 B/A 的退化分支对应零 RKHS 头，不是经验阈值选参。
5. **完整矩阵与独占输出。** 总 spec 为 8 个 model×cohort lane；每模型 rx3=900、rx1=300 个任务，共 4800。K1=1200 仅数值诊断；3600 个 OOF 任务各 3 折×3 臂，共 32400 次分解。reader 拒绝重复/缺失 cell；probe 完成标记仅在任务循环全部结束后产生。run root、row、probe、文件均在新目录中创建；已有 run/release/archive 会拒绝。lane 失败保留证据，不自动重跑，其他 lane 可完成。
6. **CPU 发布闭包。** 新 publisher 精确替换已有 transport 的 runner 并删除唯一 GPU 检查。实际 remote 脚本设置 CUDA_VISIBLE_DEVICES 为空、4 个 CPU lane、每 lane 2 个 BLAS 线程；不会调用 exporter。PATHS 覆盖 code、新旧 entry/runner、cache 常量依赖、cvs_native_artifacts、两份 summary 及三份配置，总 spec 另由 transport 添加。纯导入核验所有本地路径存在；未执行发布。transport 保留 pushed commit 检查、归档路径限制、唯一输出与实际 commit 记录。
7. **配置与日志。** 已用本地 Python 精确比较 core FROZEN_CONFIG 与 frozen/support_rx3/support_rx1 三份 algorithm，全部相等；cohort matrix 与总 spec 相等。trace 保存逐物理折、三臂 OOF、损失、梯度残差与阶段成本，compact 保留标量；既有模型包字节不冒充最小部署包，本次不加载/传输 checkpoint。仅 support 诊断，不产生正式 query prediction 或自动晋级。

## P1 与定点收束

初版 tools/summarize_d92_branch_interaction_probe.py 第 120 至 129 行由 overall 读取 H、新类平均值，只对旧类按 K 检查。于是一个 K 的负增益可被其他 K 的正增益掩盖，导致不满足预登记仍显示 PASS。预登记 prepare 第 40 行和 design 第 86 行由主任务确认为每个 K 均应通过。

修复后第 122 至 135 行对 K=5、10、20、两个对照分别检查 H>0、新类准确率差>0、旧类差≥−0.01，全部有值且全部满足才显示 PASS；overall H/new 只保留描述。已定点读回代码及反例测试：2 对照×2 指标×3 K×负值/零值，共 24 种“整体为正但某 K 不严格提高”的情形均要求 FAIL。作者运行该 summary 文件的 5 项测试全部通过。修复仅涉及判定及测试，未改公式、数据或预算，未读取新实测结果。此 P1 已关闭；未重复审查其余不变代码。

## 证据与未做事项

本次实际执行的是代码路径检查、纯导入与配置/发布清单相等性断言。作者提供的已通过证据为核心 29 项加旧相关 40 项、入口/汇总 8 项、编排 10 项；审查未重复运行这些既有测试。该证据不等于真实数据运行完成。

本次未读新 query 成绩、truth、IQ 或旧目标结果来决定设计，未启动实验、写远端、提交或镜像。support cache 的内容来自获准的既有 producer；本次没有重新执行 checkpoint、特征提取或物理数据验证。源码检查支持所述访问边界，不是对任意外部环境、历史数据来源和未来结果的无条件合规保证。K1 无 OOF 性能证据，support OOF 也不等于新独立 query 泛化。

## 定点源码

- [core](../code/cvsrffi/d92_branch_interaction.py)：95–118 行核中心化与逐行分数；123–166 行训练；242–291 行物理折。
- [新入口](../tools/evaluate_d92_branch_interaction_probe.py)：28–33 行缓存 reader；71–82 行当前 support 切片；112–119 行终态。
- [缓存 reader](../tools/evaluate_d92_branch_support_probe.py)：59–138 行，精确元数据、成员、物理 ID 和完整矩阵校验。
- [runner](../tools/run_d92_branch_interaction_probe.py)：13–37 行 spec；59–87 行独占输出及逐 lane 状态。
- [publisher](../tools/publish_d92_branch_interaction_probe.py)：5–15 行依赖及 CPU transport。
- [预登记准备](../tools/prepare_d92_branch_interaction_probe.py)及[总 spec](../configs/d92_branch_interaction_support_20260929.json)。
- [汇总判定](../tools/summarize_d92_branch_interaction_probe.py)。
