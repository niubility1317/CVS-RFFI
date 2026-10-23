# 四种Full方法正确practical residual与普通PL四格及PAMF匹配对照

- run_id：`20261023-phase1-practical-residual-full2x2-m600-r01`
- group_id：`phase1-full-practical-residual-factorial`；类别：`comparison`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

11物理合同、四种Full原生目标×信道开关×普通PL；匹配PAMF监督与新RX原生T1，三种泛化测试。

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


## 发布实现和本地验证

本次600个新F=528外部Full+66匹配监督PAMF+6新RX原生T1，另4条完整匹配G；27个既有T1只补三类泛化读出。全部逐行冻结raw/匹配G、三类泛化和七景后独立CPU评分。取消外部CE、固定L/V、chrono训练、m5和细控制扩张。

加速已实现为逐轮异步CPU producer、按L物理ID读取的共享FP32单场景缓存及下一轮预取；不缓存模型目标。多方法/种子/PL开关复用同一个确定性输入函数，重复ID/乱序消费逐元素一致。PAMF监督同时修正As和附加槽，原生T1配方不变。新G复用已验证源观测并启用既有冻结教师epoch/ID缓存与零权重FFT跳过。

10项聚焦本地检查PASS，包括route、矩阵/预算/RX、禁U/禁truth、无增强冻结读取与缓存同函数乱序消费。管理Python未安装Torch的初次环境检查不算模型失败；实际检查使用现有phase1-cu128环境。唯一独立P0/P1审查3个直接配置问题已修复并定点确认，无未解决P0/P1。真实checkpoint/source smoke是远端launch第一步之一，PASS后自动继续，当前不提前宣称通过。

用户最新要求保留三张卡：仅GPU0–4，每卡最多3、最多15个本任务GPU进程；GPU5/6/7保留。正式资源准入仍计入其它owner、显存/CPU/RAM/IO guard。没有停止、热改或重启任何旧健康任务。无已有正式新性能。

## 启动包装技术恢复

首次release在控制器导入plan前尚未接入历史Full方法namespace，ModuleNotFoundError退出；PID1906598已独立核实结束，0数据准备、0科学worker、0GPU训练、未建立run root。旧release、归档和日志保留。修复仅将bootstrap提前到导入plan/cache之前，并增加该有界导入回归测试；11项聚焦检查PASS。新不可覆盖control-r02使用同一登记600F/4G科学矩阵，保留GPU5/6/7，不重新创建第二份模型矩阵。

control-r02已完成11源合同、三种泛化句柄和ManyRx源RX/D0的31279条×六景观测，后在历史G描述外层合同字段兼容处退出。0模型worker、0GPU训练，旧数据/日志/4个未启动G配置保留。control-r03仅修复合同路径解析并复用已准备句柄，使用新输出根。新增整队列回归验证1880个配置的stage、72个PAMF的ablation字段及两种G描述格式，12项聚焦检查PASS；不重复模型全审查。

## 旧Torch执行API兼容恢复

r03真实checkpoint、同合同G-L/G-LU和四Full源侧梯度smoke已PASS；G和缓存已GPU运行。两个F在Torch1.11.0/NumPy1.24.4的精度设置API处、创建模型/输出前失败，2同指纹正确停止新派发。新r04保留健康科学worker和CPU producer原PID/CWD/config；仅替代2个失败F及各预测/评分6阶段，598未启动F只切换执行release。旧失败及阻塞记录保留排除，有效600F/4G不变。通过TF32关闭flags保持原FP32，老Torch缺少的新API仅做功能存在判断，不降精度、不改训练。复用源smoke与全部数据；新增两环境精度与输入复制检查。
