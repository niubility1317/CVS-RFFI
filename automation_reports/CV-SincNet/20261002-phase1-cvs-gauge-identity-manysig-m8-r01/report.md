# CVS 完整 IQ 相位规范化：源域预登记

状态 SOURCE_TRAINING，正式源训练运行中；新的 clean 测试尚未进行。仅普通 CE、身份骨干、无增强；8 个新 scratch 行，固定原物理 L6300/V27000，U56700 不使用。两个参考方案×四 seed，E200×50 步、batch128、AdamW2e-4/wd1e-4/cosine1e-6、FP32，无裁剪、teacher、EMA、域损失或继承权重。

峰值参考和四周期相干参考位于全部身份路径之前，完整 256 点 IQ 仅乘同一复标量。整网公共相位不变性限定实数精确运算，不消除 CFO；AM/AM、AM/PM、RX 和多径响应作为诊断，不能声称唯一辨识 TX 硬件。两方案均为 164225 参数，固定前端为 0 参数，与原残差核心按种子初始化一致。

源排名固定为 8 个新记录加 4 个历史 residual_fusion 源记录。控制来源/完整物理角色/预算已实时核实，只读源指标，不继承权重、不读目标成绩。四 seed 最终 0.5×V+0.5×最差源 RX 分数最高优先；完全并列依次 V、最差 RX、MAC、参数、残差/峰值/相干顺序。成本只是次要条件。

新方案胜出后默认启动 `20261002-phase1-cvs-gauge-clean-manysig-m24-r01`：四个新预测＋二十个旧冻结预测只读复用，同 168000 个物理 clean query/6TX/7RX，全部预测固定后独立 truth-last 评分。若残差源控制胜出，保留其已完成历史测试，不重跑历史实验，不测试未选的新方案；新方案测试记 N/A、报告源拒绝。源表现/理论属性都不能替代测试性能证明。

已通过 13 项模型/物理聚焦检查与 10 项选择/输入负测。第一次物理角色负测因测试 fixture 浅拷贝使期待原件同时改变而失败，已改为深拷贝并重测 10 项 PASS。模型独立 P0/P1 审查 PASS；执行/协议独立 P0/P1 审查 PASS，历史源字段与校验器兼容。远端 Torch2.1 运行 smoke、真实曲线、GPU 成本及 clean 新结果尚为 N/A。保护所有健康任务和历史产物。

[前瞻结构与数学边界](../../../docs/CVS_GAUGE_IDENTITY_HYPOTHESIS_20261002.md) · [源控制实时证据](evidence/source_control_preflight.json)。

## 发布后独立读回

VERIFIED：提交 `8f20a4e8dc7380f75a3966ac04fa282c2f110148` 的远端 OID 一致；归档传输校验、一次远端 compile、PyTorch2.1 的八个一次性 CPU 模型和24次 CE 更新均通过。dispatcher PID 1177298，八行各占一块 GPU；CWD/argv/PID/resolved/日志增长已独立核实。新源训练仍在运行，不报告最终指标或性能目标完成。[读回证据](evidence/source_launch_readback.json)。

clean 链路实现已通过47项相关检查及独立 P0/P1 审查：重新核对8新+4旧源记录，重算排名；基线胜出时拒绝新query，选中gauge才4新+20旧统一truth-last。当前源训练继续运行，尚未冻结/生成真实clean矩阵/访问新query。
