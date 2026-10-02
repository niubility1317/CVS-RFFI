# Group完整query原始评分入口验证

2026-10-02，状态VERIFIED_LOCAL_WRAPPER_AND_PLAN_NOT_DISPATCHED。

原Group运行已在2026-10-01T23:51:44Z之前完成全部4 row/2400任务，终态依据原run的独立metadata读回。新的固定SupportMetric公式及完整query矩阵已在读取任何当前query成绩前登记；评分结果禁止回流调参、选择或重跑。

tools/execute_frozen_d92_group_query_score.py仅使用stdlib，在读取原spec/startup/complete并闭合原runtime73aa、完整矩阵及已固定预测后，只调用原不可变release中的scorer。原scorer自行独立校验并重读全部预测后才连接truth；wrapper不拟合、不载入模型、不读summary或声明评分成功。独立输出目录为原run加-score-r01，拒绝覆盖和自动重试。

root串行激活ssr-gpu后执行35项离线用例，全部通过，Python3.10、NumPy2.2.6、SciPy1.15.3、Torch2.1.0+cpu；无Conda CLI或hooks。完整stdout/stderr保留gzip，见automation_reports/CV-SincNet/20261001-phase2-d92-group-barrier-joint-repeat-m2-r01/evidence/original_score_wrapper_validation_20261002。测试验证严格runtime/路径/矩阵绑定、独立目录、普通用户、唯一scorer argv、进程异常保存及Linux wait4子进程RSS口径；合成测试没有访问真实数据或SSH。

wrapper只声明PROCESS_EXITED_AWAITING_ARTIFACT_READBACK；真实评分完成必须由root另一次metadata/artifact读回核实。此提交阶段尚未dispatch，实际成绩N/A。新SupportMetric support已完成全4/160/1800，独立分析仍是原登记的r01未发布；未根据其成绩改方法。
