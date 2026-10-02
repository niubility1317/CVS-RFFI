# CVS 因果包络耦合：源选候选clean确认预登记

状态LOCAL_VERIFIED，尚未访问本轮新query。8个新scratch模型完整E200×50，加4份固定energy源控制，按原性能优先规则选择`coupled_lag4`，测试前全部冻结。仅4份新预测，复用24份原固定控制，共28行；相同168000物理ID/6TX/7RX，全部预测固定后由独立scorer连接truth。

普通CE、无增强、身份骨干、原划分，固定E200。原读出/深宽不变，202553参数，实际lag与完整模型/source/payload/physical roles/scratch/fullFP32契约在query前核对。69项相关检查及独立P0/P1审查PASS，不重复数据验证和旧控制预测。

新源与预测完整FP32，旧控制保持原历史精度。仅clean，不测试未选候选或追加LEO/SFT/support/新类；测试结果不反馈lag、结构、超参数、候选排序或选择性重跑。设计延迟及常相位性质不等于唯一TX硬件参数恢复，尚无本轮识别提升结论。

[完整源报告](../20261002-phase1-cvs-coupled-identity-manysig-m8-r01/report.md) · [源冻结](evidence/performance_selection.json) · [路径验证](evidence/conditional_clean_local_validation.json)。
