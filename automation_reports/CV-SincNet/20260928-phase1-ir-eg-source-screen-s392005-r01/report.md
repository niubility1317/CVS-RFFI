# IR-EG源侧筛查已发布

发布状态：**VERIFIED**。N607六行训练的PID、CWD、argv、CUDA绑定、实际配置及接受步日志增长已独立核实。训练仍在进行，尚无完整结果。

固定model seed392005，split/data/augmentation/evaluation seed392005；support不适用。六行均从零训练，L/U/V为6300/56700/27000，同一既有物理ID契约，完整原生DAOT＋FastTrust。200epoch/44400接受步是固定曝光预算，不代表收敛。

|行|GPU|PID|读回接受步|状态|
|---|---:|---:|---:|---|
|SIM_s392005|1|1946779|99|RUNNING|
|EG_s392005|3|1946780|68|RUNNING|
|IR_s392005|4|1946781|63|RUNNING|
|IR_G0_s392005|5|1946782|58|RUNNING|
|OR_EG_s392005|6|1946783|64|RUNNING|
|IR_ENCODER_OFF_s392005|7|1946784|62|RUNNING|

读回时间：2026-09-28T00:43:56.158891+00:00。发布commit：`92f40367d61c45b9d16a9718bcd68df17af507c2`。归档SHA：`adacb95e57cc53835b71b433bb7ba370833170f0dd3bddd68cd43ac961f99e8b`。

## 验证

- 本地旧PyTorch接口兼容专项2项通过；独立P0/P1审查发现的autocast签名问题已修复并定点闭合。
- N607实际torch2.1.0+cu121 CUDA环境全部112项IR测试通过，包含active IR路径；单次归档校验及远端编译通过。
- 原生scratch模型checkpoint保存、CPU反序列化、严格恢复及4个合法L_s样本前向检查PASS。该诊断checkpoint没有被训练继承。
- 新建run/log根目录，六个GPU各一行，GPU0/2原有任务保持运行。

## 环境准备与边界

初次测试调用因N607环境没有pytest失败；远端PyPI超时，随后使用本地下载的pytest8.3.5及依赖离线安装到独立/tmp测试目录。未修改现有Conda环境，训练不使用该测试PYTHONPATH。

本次仅启动六行单seed源侧筛查。三seed确认、BR效率和目标预测尚未启动，依设计待源侧证据和候选冻结后推进；收敛延长配置仍未冻结。本次发布不形成性能、收敛或目标泛化结论。技术异常只影响所属行，不自动重试、跳过坏batch或按性能停止。

## 证据

- [发布登记](experiment.json)
- [112项远端测试](evidence/remote_acceptance.log)
- [归档与编译](evidence/publish.log)
- [提交进程](evidence/launch.log)
- [最新独立读回](evidence/readback_1790556295011623100.log)
- [发布审查](../../../experiments/ir_eg_v1/docs/release_review_20260928.md)
