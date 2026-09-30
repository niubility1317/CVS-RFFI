# LORA真实WiSig源代码完整链路验收

- run_id：`20260930-diagnostic-lora-wisig-s392005-r01`
- group_id：`newclass-registration-source-acceptance`；类别：`diagnostic`；阶段：`external_execution_acceptance`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

固定作者逻辑、从零训练，各阶段1epoch，保留原始算法行为；执行验收而非论文准确率或正式CVS比较

## 数据、seed与模型来源

实际数据契约、权限例外、完整seed角色、checkpoint来源和选择规则见experiment.json。
逐行配置通过config_ref/resolved_config_ref定位；待补项必须在对应生命周期补齐。

## 执行与存储

命令、环境、CWD、commit、launch owner、输出和日志路径见experiment.json。
实际PID/GPU、读取时间、remote readback、失败或替代关系在此追加，并用record命令记录证据指针。

## 结果与覆盖

尚无结果。按预登记artifact逐项记录路径和缺项；保留每row与RX/day/TX/scene/K/seed的对应关系。
源域训练完成、预测完成、评分完成及协议有效性分别陈述。不得用总索引或旧状态证明当前运行。

## 交接

记录已完成、当前run/commit、证据路径、阻塞与下一步；恢复先查原run，不重复启动。

本次范围：真实WiSig执行诊断，不等于原论文数值复现或正式CVS对比。使用本次scratch状态，没有外部权重继承。详细逻辑/修改见 paper_reproduction/newclass_registration_20260930/README.md。独立P0/P1审查及定点复审通过。

## 真实WiSig验收结果（VERIFIED）

6个TX、RX1-1、2021_03_01、raw IQ；旧3类、新3类，K=128训练样本/类，64个测试样本/类；每阶段1epoch。训练/测试物理ID不相交；scratch初始化，teacher/key仅继承本run。源验证N/A，无query反馈选模，未使用LEO。

|阶段|旧类准确率|新类准确率|H|
|---|---|---|---|
|A：预训练/注册旧类|97.9167%|N/A|N/A|
|B：仅旧类support适应|97.9167%|N/A|N/A|
|C：新旧类统一竞争|97.3958%|96.3542%|96.8722%|

B=A：冻结提取器、相同旧类support与同域数据没有参数适应。适应提升0个百分点；注册后旧类下降0.5208个百分点，新旧差1.0417个百分点。单RX/同日/短预算结果不能推断跨域泛化，也不满足正式CVS新类LEO口径。作者原长度预处理最大误差1.616547e-7；模型为第三方Torch端，完整训练数值等价仍未证实。

全部12训练步的loss/梯度有限；384个唯一物理ID预测齐全；1个checkpoint均strict重载并前向验证。详细文本、每步JSONL/CSV与epoch汇总见evidence目录。

硬件：RTX5070Ti，torch2.10.0+cu128；方法端到端运行24.6021秒（包含预处理/训练/保存/预测，不含独立评分），Torch峰值allocated=43555328字节。模型参数=367104；LoRa训练时全部模型参数可训练，注册阶段0个参数更新。逐阶段训练时间见steps.jsonl的summary行。独立推理耗时、主机峰值内存、完整常驻状态与地面新增传输量未单独测量，记N/A；不宣称星载省算力。

原论文复现尚未完成：ISSL全文与原始数据未获得；LoRa作者TF2.1环境/原数据尚未完成。正式CVS完整矩阵未启动。这里只验收授权的真实WiSig运行，不因低准确率改超参或选择性重跑。
