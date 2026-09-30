# ISSL全面验收两轮真实WiSig回归

- run_id：`20260930-diagnostic-issl-wisig-s392005-r02`
- group_id：`newclass-registration-source-acceptance`；类别：`diagnostic`；阶段：`external_execution_acceptance`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：ANALYZED（2026-09-30；独立产物读回与全日志分析VERIFIED）

## 目的与对照

修复输入/评分/契约路径后完整2epoch功能回归；checkpoint重载预测一致，原论文不使用WiSig时不启动完整预算复现实验

## 数据、seed与模型来源

实际数据契约、权限例外、完整seed角色、checkpoint来源和选择规则见experiment.json。
逐行配置通过config_ref/resolved_config_ref定位；待补项必须在对应生命周期补齐。

## 执行与存储

命令、环境、CWD、commit、launch owner、输出和日志路径见experiment.json。
实际PID/GPU、读取时间、remote readback、失败或替代关系在此追加，并用record命令记录证据指针。

## 结果与覆盖

验收结果、逐阶段指标及缺项见下方“全面执行验收结果”；完整证据保存在evidence目录。
源域训练完成、预测完成、评分完成及协议有效性分别陈述。不得用总索引或旧状态证明当前运行。

## 交接

记录已完成、当前run/commit、证据路径、阻塞与下一步；恢复先查原run，不重复启动。

范围：每阶段2epoch真实WiSig功能验收，非论文准确率或正式CVS对比。条件判断与40项测试见 paper_reproduction/newclass_registration_20260930/COMPREHENSIVE_ACCEPTANCE.md。

## 全面执行验收结果（VERIFIED）

固定旧3类、新3类，K=128训练样本/类、64个query/类；RX1-1、2021_03_01、raw IQ。每阶段2epoch；scratch初始化，teacher/key仅继承本run。物理ID不重复且训练/query不相交。无源验证，无query反馈选模，未叠加LEO；非正式CVS性能矩阵。

|阶段|旧类准确率|新类准确率|H|
|---|---|---|---|
|A：适应前旧类|66.6667%|N/A|N/A|
|B：仅旧类support适应|N/A|N/A|N/A|
|C：新旧类统一竞争|34.8958%|0.0000%|0.0000%|

独立增量对照C：旧类75.5208%、新类0.0000%、H=0.0000%；该分支从本run的base独立继承，不继承SSL/transfer状态。B不存在，适应提升及注册后旧类下降均N/A；主分支最终新旧差34.8958个百分点。两轮新类准确率为0，尚未证明有效的新类学习或收敛。

保留并通过数值测试确认原作者KD学生断梯度、SSL跨步梯度累积、queue reshape行为。执行验收通过不代表这些行为符合论文算法；全文未取得，算法一致性验收仍未完成。

40项针对性测试通过，独立P0/P1检查无阻断项。实际66训练步，全部loss/梯度有限；384个唯一物理ID预测齐全；4个checkpoint strict重载，恢复后的完整预测逐条相同（A, C, C_incremental_baseline）。训练数据契约路径存在且指向本次实际输入。

完整读取74行JSONL、74行CSV、76行stdout；无错误标记。逐stage/epoch的loss、梯度范围和摘要见evidence/full_log_audit.json。两轮只能证明跨epoch执行，不能判定收敛。

硬件RTX5070Ti，torch2.10.0+cu128。方法端到端18.2867秒（预处理/训练/预测/保存，不含独立评分），Torch峰值allocated=520555520字节，模型参数=3847424。ISSL模型全部参数可训练。阶段耗时见steps.jsonl。独立推理耗时、主机峰值内存、完整常驻状态和新增传输字节数未单独测量，均N/A；不据此宣称星载省算力。

原论文WiSig完整预算复现实验未启动：CSIL、LoRa_RFFI、MoPC-HR全文的数据集条件不满足，ISSL公开摘要未提WiSig且全文待核实。ISSL原数据/全文、LoRa原数据/作者Keras环境仍缺。r01及历史产物保持原样，不因评分调参或重跑。
