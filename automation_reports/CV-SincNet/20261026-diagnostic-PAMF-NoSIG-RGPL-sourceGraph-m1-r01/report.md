# 去 SIG_TIME 的双图伪标签源质量诊断

状态：**RUNNING，启动已独立核实**。本行使用单个学生 seed2026092701 的冻结新 NoSIG E30，尚无图质量、候选训练或目标集结果。原目标未达成。

执行 commit：`ba7efad5dee5820a392cdfc241701a228e366b10`；一次 archive 传输比对 VERIFIED、远端编译 PASS。控制器 PID104597、worker PID104898、GPU4 的 PID/CWD/argv、实际 GPU 驻留和进度增长已从独立后状态核实。真实 NoSIG E30／原默认头／固定 G 的源 L2 smoke PASS，逻辑 F4/G2、零 optimizer、零 U 真值与目标访问。

原 L6300/U56700/V27000 提取 raw 和12份恢复视图聚合特征。只有 L/U 为63000个图节点，只有 L 真标签作传播种子；按来源 RX 分组传播后排除自身 RX 的种子贡献。V 独立扩展，不能更新源图或其他 V。

固定困难池为冻结 raw confidence<0.95。raw/restored 两图同类、各自 probability≥0.95、各图至少三个其他 RX 来源支持才认证。先保存全部认证预测，再计算 V 质量。**新增 V≥200且 precision≥99.5% 才具备训练可行性；失败保留负结果，不调整 k／阈值直到通过。** U 正确率未知。

图核心和权限边界5项检查、局部编译及唯一独立 P0/P1 审查 PASS。它们只说明实现具备执行条件，不能证明科学增益。NoSIG 匹配控制继续原预算，未因该诊断更改。

完整参数、来源、成本和停止规则见[预登记](preregistration.md)与[实验登记](experiment.json)，启动证据见[后状态](launch_evidence.json)。方法见[PAMF-RGPL](../../../docs/goals/PAMF_NOSIG_RGPL_DESIGN_20261026.md)，复盘见[完整历史结果](../20261026-analysis-PAMF-SSL-retrospective-r01/report.md)。
