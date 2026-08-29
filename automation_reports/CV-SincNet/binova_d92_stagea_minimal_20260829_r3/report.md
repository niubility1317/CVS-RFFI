# BiNOVA-D92阶段A最小实验报告（r3技术修复）

- 状态：`LOCAL_VERIFIED`
- run ID：`binova_d92_stagea_minimal_rx20_1_s713101_20260829_r3`
- r2差异：仅统一Torch→NumPy显式dtype重建；科学方法、矩阵、门槛、场景、split、seed和资源预算不变。
- 阶段A达门槛后自动继续阶段B，否则记录`NOT_RUN_GATE_NOT_MET`。

## 运行结果

- 阶段A A0/A2/A3/A4均完成；四行伪注册H均为91.667%，A3相对A2提升0，故门槛未通过，阶段B为`NOT_RUN_GATE_NOT_MET`。
- A3非仿射残差90.28%，forgetting和旧类floor均未恶化。
- 首份prediction因场景级split与权威new2 data handle的split不一致，被独立scorer拒绝；新增splitfix配置后用新目录重预测，不修改原prediction。
