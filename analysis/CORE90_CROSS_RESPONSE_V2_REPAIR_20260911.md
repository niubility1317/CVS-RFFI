# V2复审缺陷修复与独立复核

日期：2026-09-11。授权：用户“继续优化修复，然后审查”。基线b80eed751bcd89e69c49ec7dff01e530aeb4612b；依据[前次复审](CORE90_CROSS_RESPONSE_V2_REAUDIT_20260911.md)和原22项实施计划。沿用隔离V2工作树；本轮仅代码修复、合成验证与审查，没有启动正式E200/N607实验，也未进行target再评分。

## 修复结论

|ID|对应要求|本轮结果|证据与边界|
|---|---|---|---|
|R1|V205/V216，完整基础更新→源观测→三条件门|VERIFIED：真实trainer调用源审计，成功commit后累积配对观测并驱动下一步门|完整实际loss图包含当步CORE90已激活项；复制相同AdamW/aux/scaler，A基础、B基础+响应，同加法/unscale/clip顺序；不是CE+orth近似|
|R2|V211，可靠反馈必须当前授权|VERIFIED：拒绝静态未授权、U1/head_only/永久detach或错误契约；绑定当前源契约/联合目标/范围/实时门|门关闭时uniform；独立审查发现的phase首批旧资格已提前到loader迭代前撤销|
|R3|V213，delta_pairs竞争类产物|VERIFIED：缺失、未冻结、类别结构非法等初始化拒绝|delta模式保留全部竞争类；本轮未伪造或冻结真实噪声/关键类对参数|
|R4|V220，非有限事件强制诊断|VERIFIED：响应输出、完整loss、主梯度和辅助事务异常触发诊断|非诊断步也记录；原有事务/GradScaler策略继续负责跳步与隔离|
|Review|修复后独立审查|VERIFIED：未发现新增P0/P1；新增1项P2已修复并独立定点测试通过|[独立记录](core90_v2_repair_evidence_20260911/independent_review.md)|

关键代码：[训练接线](../code/SSDG/train_ssdg.py)、[运行时](../code/cvsrffi/cross_response/integration.py)、[实际配对更新](../code/cvsrffi/cross_response/source_mechanism.py)、[反馈资格](../code/cvsrffi/cross_response/scheduler.py)、[决策产物验证](../code/cvsrffi/cross_response/decision_calibration.py)。

## 验证

相关回归：189 passed、12 skipped；跳过项为环境控制的入口测试，真实CUDA入口由独立合成脚本执行。[原始pytest日志](core90_v2_repair_evidence_20260911/pytest.log)。新增恢复验证保存1/2配对窗口，恢复后下一批模型、辅助头、optimizer、scaler、采样器、RNG及门观测逐项一致。实际AdamW单元对照覆盖无clip及clip=0.05，A/B参数与optimizer状态完全匹配正常更新实现。

|真实CORE90合成入口|批次|成功主更新|有效配对更新|门观测|对照事件数/支|主轨迹一致性|
|---|---:|---:|---:|---:|---:|---|
|mechanism_fp32_01|4|4|4|2|20|VERIFIED|
|mechanism_amp_scale128_01|8|8|8|4|40|VERIFIED|
|mechanism_amp_default_03|8|3|2|0|40|VERIFIED|

对照覆盖采样物理ID、前向结果、完整基础loss、未缩放梯度、主模型/AdamW/EMA/prototype/pseudo状态及RNG。AMP非有限值比较精确mask和符号，不把NaN相等作为数值接近。FP32在phase提前撤销补丁前完成，测试中门始终关闭；该补丁由新增定点测试和补丁后的AMP入口验证。

默认AMP最初2轮、4轮验证因不能形成两次有效配对观测而触发断言，均保留[失败明细](core90_v2_repair_evidence_20260911/initial_amp_failures.json)。实际发现前三步默认scale溢出跳步；后续存在主更新成功、反事实B溢出的情形，审计正确丢弃窗口并关闭门。默认scale最终单独验证主轨迹一致和失败关闭，不声明有效观测成功。可用观测路径另用两支相同的合成init_scale=128验证；这是显式测试设置，不修改生产默认scaler。

每个配对更新都读取独立源V，同一物理查询在A/B上使用clean+三LEO输入，逐样本面对全部类；报告保留每组风险、配对差值、标准误和审计成本。标准误的统计单位是配对训练批次，不是独立训练seed。完整张量对照留在本机logs，Git仅收录可检查的JSON/JSONL/文本日志，无IQ或checkpoint。

## 仍然适用的科学边界

- 当前规范配置仍未冻结真实source gate阈值、拟合预算、可靠反馈证据和delta/竞争类产物；无合格证据则身份响应保持关闭。本轮验证工程可达性和无副作用，不证明源质量、跨接收机收益或晋级。
- head_only必要性对照是当前冻结特征上的匹配临时头拟合，并非独立训练的head-only骨干。不能由该观测推断完整head-only训练对照收益。
- 反事实只模拟一次实际student AdamW更新的风险；当步EMA/proto/pseudo输入已在完整loss图内，但不模拟反事实的后续EMA/proto演化或多步训练。
- training_physical_ids字段只列响应块有标输入；真实pseudo基础图还可能读取U_s。安全依据是初始化完整L_s/U_s/V角色互斥，不能将该字段当作完整基础图输入清单。
- V218历史训练分叉的首个分歧仍未定位。新合成主轨迹一致不等价于解释历史分叉；历史UNRESOLVED状态保留。

本轮缺陷R1—R4及独立新增phase时序问题已关闭；真实源证据冻结、历史分叉解释和正式实验收益属于仍未完成的科学证据，不升级为通过。
