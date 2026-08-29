# BiNOVA-D92阶段A最小实验报告（r3技术修复）

- 状态：`LOCAL_VERIFIED`
- run ID：`binova_d92_stagea_minimal_rx20_1_s713101_20260829_r3`
- r2差异：仅统一Torch→NumPy显式dtype重建；科学方法、矩阵、门槛、场景、split、seed和资源预算不变。
- 阶段A达门槛后自动继续阶段B，否则记录`NOT_RUN_GATE_NOT_MET`。

## 运行结果

- 阶段A A0/A2/A3/A4均完成；四行伪注册H均为91.667%，A3相对A2提升0，故门槛未通过，阶段B为`NOT_RUN_GATE_NOT_MET`。
- A3非仿射残差90.28%，forgetting和旧类floor均未恶化。
- 首份prediction因场景级split与权威new2 data handle的split不一致，被独立scorer拒绝；新增splitfix配置后用新目录重预测，不修改原prediction。

## splitfix1最终结果

- 状态：`ANALYZED`；160条query的四状态prediction完整，独立scorer确认`truth_join_after_prediction_only=true`。
- 阶段B：`NOT_RUN_GATE_NOT_MET`。

|状态|旧类准确率|旧类floor|新类准确率|H_old_new|
|---|---:|---:|---:|---:|
|DA0_REG0|90.833%|80.000%|N/A|N/A|
|DA1_REG0|93.333%|80.000%|N/A|N/A|
|DA0_REG1|92.500%|80.000%|100.000%|96.104%|
|DA1_REG1|90.833%|75.000%|100.000%|95.197%|

- 域适应效应：注册前旧类准确率+2.500个百分点；注册后−1.667个百分点，旧类floor−5.000个百分点。
- 交互项：旧类准确率−4.167个百分点，旧类floor−5.000个百分点。
- 结论：A3的REG0收益未迁移到注册竞争，不晋级阶段B；支持集门槛正确节省了后续训练。
- r3主训练约70秒，GPU0新增进程观测显存约684MiB。
