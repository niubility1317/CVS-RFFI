# CVS 可学习相位记忆残差：源选候选clean确认预登记

状态LOCAL_VERIFIED，尚未访问本轮新query。8个新scratch模型完整E200×50，加4份固定coupled_lag4源控制，按原性能优先规则选择 `adaptive_volterra_lag4`，测试前全部冻结。仅4份新预测，复用28份原固定控制，共32行；相同168000物理ID/6TX/7RX，全部预测固定后由独立scorer连接truth。

普通CE、无增强、身份骨干、原划分，固定E200。原读出/深宽不变，202555参数，比原控制多2个全局CE训练门，实际phase lag及固定envelope lag4与完整模型/source/payload/physical roles/scratch/fullFP32契约在query前核对。新增30项公共clean路径/完整分析检查、旧70项检查和独立P0/P1均PASS，不重复数据验证或旧控制预测。

新源与预测完整FP32，旧控制保持原历史精度。仅clean，不测试未选候选或追加LEO/SFT/support/新类；测试结果不反馈延迟、系数、结构、超参数、排序或选择性重跑。输入lift仿射相位协变和整网共同相位性质，不等于任意CFO/RX/LTI不变或唯一TX硬件参数恢复。尚无本轮识别提升结论。

[完整源报告](../20261002-phase1-cvs-adaptive-volterra-identity-manysig-m8-r01/report.md) · [源冻结](evidence/performance_selection.json) · [本地验证](evidence/conditional_clean_validation.json) · [独立审查](evidence/conditional_clean_review.json)。
