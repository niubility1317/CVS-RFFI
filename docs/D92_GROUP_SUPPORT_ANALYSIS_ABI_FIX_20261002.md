# GroupBarrier support分析引用格式修复

修复仅涉及独立分析器读取接口。support训练runtime9518d46d2与query预测runtime73aa1b31f不变，未重新拟合、调参数、改预测或读取任何成绩。

## 已核实原因与修复

独立分析r01于141.123秒后失败，未产生summary或任何表格。证据为[原失败读回](../automation_reports/CV-SincNet/20261001-phase2-d92-group-barrier-joint-support-m2-r01/evidence/analysis_runtime_1790872597649626100.json)。`compact_event`对嵌套audit/preparation引用调用生产`scalars`，该函数省略list值，因而紧凑日志中的引用缺少shape；canonical StateArchive和NPZ保持完整。旧分析器误把紧凑投影当完整descriptor，触发`KeyError: 'shape'`。这是读取格式错误，不是训练失败或数值状态缺失。

正式trace与完整training_events仍要求完整shape/dtype/nbytes与已注册canonical descriptor一致。仅fit_stages.jsonl中的紧凑引用允许生产投影，且全部字段必须恰等于canonical StateArchive或实际Recorder完整描述的真实scalars投影；未知、混合、篡改投影显式失败。没有自动补shape、忽略引用或放松manifest/NPZ检查。

## 实际验证与新执行范围

根Agent使用已核实ssr-gpu原生环境执行受影响分析器完整文件，20项合成case全部通过（`pytest_native_activation_1790873459909748300`）。新增9项覆盖真实Recorder/StateArchive/compact_event嵌套引用、两个完整流缺shape负测，以及混合shape、dtype、nbytes、summary、额外字段和未知path投影负测。包含原完整四行160parent合成闭合。原r01失败与此前11项验证记录保留，不据测试通过声称真实分析已经完成。

修复源码push并独立核对OID后，按同一原run预登记一个新分析attempt r02，独占新release/output；不覆盖r01，不自动重试原代码，不更改或干预正在运行的query任务。新分析只读同一完整四行原产物，完整结果闭合后才报告support OOF/proxy。真实query依然等待完整固定A/B/C和独立truth-last。Goal仍ACTIVE。
