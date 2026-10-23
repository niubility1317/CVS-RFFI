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

## 实际发布核验

状态 **VERIFIED / RUNNING**。实际执行 commit `dd2f4bfb389a9ff684fcb0d071d194543699d895`，owner PID 1925715，CWD/argv和物理GPU独立核对通过；旧r03 controller正常交接退出，健康G与CPU增强producer保留原PID/CWD/config。

两次观测间隔 193.9 秒，9 条相同 PID、9 条日志增长，其中 6 条为实际外部 Full F 训练。11合同已准备，真实checkpoint/source smoke PASS，U真值/target读取0。当前本任务 9 个GPU worker；GPU0–4分别为 {2: 1, 1: 1, 0: 1, 4: 3, 3: 3}，5/6/7保留，其他owner不动。无未解决失败。

有效阶段状态：`{"RUNNING": 9, "QUEUED": 1870, "COMPLETE": 1}`。600新F+4完整G保持，27既有T1只补读出；1880逻辑阶段、627行冻结预测/独立评分。14项聚焦检查和限定P0/P1/直接修复复核通过。尚无正式新性能或整链加速倍数；加速仅按后续真实吞吐校准。
私有逐进程证据保存在 local_artifacts/external_practical_residual_2x2_design_20261023/verification_1792746688.json；公开摘要为 launch_verification.json。

## IB-RFF 第二阶段作者导入恢复 r05

一次小时检查发现两条 F0-ibrff-A（seed2026092701/02）在 stage2 首个 HSIC 损失计算前失败，指纹均为 `ModuleNotFoundError: No module named 'utils.hsic'`，已按规则停止新派发。源侧复现确认 G 的常规 `utils` 包遮蔽 IB 作者无 `__init__` 的同名目录；不是数据或数值机制变更。

修复只按已登记作者路径独立加载原 `hsic.py`，不替换全局 G utils、不改 HSIC 实现或0.04权重。Torch1.11真实同seed第一阶段 checkpoint与12条合法L样本检查 loss/gradient 与直接作者函数逐元素一致，target/Utruth/optimizer update均0。16项聚焦测试PASS，限定独立P0/P1恢复审查PASS，无未解决P0/P1。

新不可覆盖 control-r05只替代2失败F和4阻塞预测/评分，共6阶段；2条失败F保持同scratch配方/seed/最大预算从零恢复，旧24/28轮stage1、2600更新及约250.31共享worker秒作为技术失败部分成本保留，不混入正式模型成本/评分。130条未启动IB F只换执行release，其它健康、已完成或未启动非IB均不改。原6个完整F、2恢复IQ缓存、4条健康G和producer原PID/CWD/config保持。

科学矩阵仍600新F+4完整G、627读出、1880有效阶段；历史1892阶段、12技术排除。新旧映射由current jobs.json/technical_recovery.json及previous_recovery指针确定。GPU0–4且每卡总compute<=3，5/6/7继续保留，不占其它owner。发布前不能称已恢复GPU训练；实际结果需独立PID/CWD/argv/GPU/log增长核实。

## SDG 原作者片段打乱 NumPy 兼容恢复 r06

两条 F0-sdg-A（seed2026092701/02）在原作者 `permutation` 的 `np.random.permutation(splits)` 处失败。256点分为3段时长度不等，NumPy1.24不再自动将该list转换为object array；源侧与局部回归均复现，2同指纹触发停新派发。没有完整训练轮日志，旧模型/输出/失败日志全部保留，不能当正式负结果。

兼容修复保留作者permutation的函数code和其余6个算子，仅该函数的局部NumPy代理对不等长list显式构造原object array，再调用原RandomState.permutation。全局NumPy不改。equal/random两模式×40seed共80例，与按片段索引排列的参考实现逐元素输出及完整RNG状态一致。实际Torch1.11/NumPy1.24、12条合法源L、作者AugDataset两原生视图及Full默认头联合更新一次（disposable smoke）PASS，正式训练更新0、target/Utruth读取0。17项聚焦测试及一次限定直接问题P0/P1审查PASS。

新不可覆盖control-r06只替代2失败SDG F+4阻塞消费者6阶段，130未启动SDG F只换执行release。原G、健康IB、producer保留原PID/CWD/config，已完成F和恢复缓存不重跑；此前HSIC已在真实stage2验证修复。三次技术替代链各6阶段保留排除，历史1898、有效1880，原600新F/4完整G和627读出不变，预算/数据/seed/增强概率不变。只用GPU0–4，每卡总compute<=3并计入其他owner，5/6/7保留。必须独立核实真实SDG轮次/日志增长后才称本次恢复完成。
