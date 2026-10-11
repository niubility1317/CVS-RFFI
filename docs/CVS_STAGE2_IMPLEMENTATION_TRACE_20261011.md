# 第二阶段多解耦实现追溯

设计依据：`CVS_MULTI_DISENTANGLE_STAGE2_20261011.md`。用户已授权实现并发布，覆盖原设计报告的“仅设计”范围；测试明确只测 clean。固定 C0/C1/C2/C3 × 三种子，E40 × 222 次更新，每 GPU 合计最多 4 个实验进程。

| ID | 设计条款 | 实现位置 | 验证要求 | 当前状态 |
|---|---|---|---|---|
| D01 | §2 合规同 seed 纯 CE 父权重，先加载后辅助构造 | common.py/source.py/data.py | 父来源读取证据；data_checks 的角色/顺序/预处理与 U 零读取 | implemented；发布前远端严格加载再次核实 |
| D02 | §5.1 独立 L/T，有界真实 IQ，分别两遍 L_s，随后冻结 | actions.py | stage2_actions_cpu_checks.json / stage2_actions_cuda_checks.json | verified；CUDA 拟合重复误差 <1e-6 |
| D03 | §5.2 条件组均值、双向 Huber、三折 TX 留出 | receiver.py | receiver_checks：9 项，留出 TX 不影响拟合/标准化 | verified |
| D04 | §5.3 单向 R 裕量修复、c 单次使用、固定关系分母 | receiver.py | receiver_checks + stage2_runtime_cuda_r03.json；实际 R→E 范数 1.834516 | verified；synthetic 梯度证据不是成绩 |
| D05 | §6 CE/keep/aug/drop/R 与固定权重日程 | source.py | runtime_checks 的手算损失、停止梯度、共享 8880 步采样 | verified |
| D06 | §6 E40/8880、两组 LR、新 AdamW、最终权重 | source.py | runtime_checks：参数覆盖、LR 边界、L/T 各1110次、完整CUDA一步 | verified；正式完整预算由结束标记核实 |
| D07 | §7 源 V 只读、8 源压力、详细日志 | source.py/data.py | data_checks；既有只读源验证；epoch/step、CUDA分段耗时、原始/加权梯度与父参考轨迹 | implemented；等待正式训练完整日志 |
| D08 | §8 12 行固定测试、clean-only、truth-last | evaluate.py/dispatch.py | stage2_control_checks_r03.json：完整合成 12×168000 预测、168＋48 条评分 | verified；实际 clean 结果待训练完成 |
| D09 | 发布与资源 | publish.py/registration.py | control_checks 的容量/独占输出/源冻结边界；远端父 metadata 只读核实 | implemented；待 Git/发布独立读回 |
| D10 | 用户追加历史回顾 | CVS_MULTI_DISENTANGLE_RETROSPECTIVE_20261011.md | 已核对历史四 seed 配对报告、7800 个源 epoch 汇总，所有文档引用检查 | verified；纯 CE 后续目标缺项明确 N/A |

## 固定工程细节

设计写有分数延迟，但旧 T 实现只有相位/CFO/缓慢相位噪声。为忠实落实该项，本版新增固定 ±0.25 sample 分数延迟：两侧各补 256 个零，以 FFT 相移后裁回 256 点。此项是新声明的小幅时序干预，不能冒称旧范围已经校准。检查零延迟恒等、正负方向、内部复正弦误差、边界与有限梯度；信号截窗之外假设为零，边缘可能有 sinc 振铃。

未指定的辅助优化器细节固定为 L/T AdamW lr=2e-4、wd=1e-4；R AdamW lr=.002、wd=0、每折 200 次完整组均衡更新。这些值在目标测试前固定，不作测试反馈调整。

独立 P0/P1 审查已关闭：参考响应融合门控学习率分组、源缓存角色实比字段、真实 R 单项至身份编码器的梯度探针均已修复并核实。未发现未关闭的 P0/P1；不构成准确率收益或 84% 达标证明。

全部程序位置均相对于 `experiments/cvs_stage2_decouple/`，测试证据位于本工作树 `local_artifacts/`，发布后正式证据登记在对应 run 的 `evidence/`。当前没有主动延后或删除任何本轮机制要求；正式训练和真实测试仍须由产物证明。
