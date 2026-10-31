# 接收上下文调制多尺度先验：从零单阶段30轮

- run_id：`20261031-phase1-DURA-contextFiLM-E30-s2026092701-r01`
- group_id：`phase1-dura-received-global-prior-context-E30`；类别：`cvs`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

原SIG-only物理与特征损失不变；27D逐record先验输入统计指导两prior四encoder层有界FiLM，初始完整G严格匹配；281415参数，完整七景自动truth-last。

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


## 固定接收上下文调制假设

原先验已接收received/base/current/normalized innovation/logvariance共9通道，不能称完全缺少状态信息。新结构只从当前同record这9通道的均值、能量和相邻差分能量取27D全局统计；每个prior独立MLP27→32→192，调制四encoder层（12/16/28/40）的scale=1+0.1tanh、shift=0.1tanh，然后继续编码和解码。不使用RX/day/scene/state/TX/F作为G输入，不声称恢复了真实硬件特质或真实完整信道响应，也不新增状态标签监督。

MLP最后一层零初值，CPU专用seed=model+2026103110+priorindex在fork中初始化，不改变CPU/CUDA probe随机流。原G全部参数名、数值、共享alias与初始完整IQ严格匹配SIG-only原对照；新增14464，总281415参数。实际合成图ConvLinearMAC25082088，对照25068072；统计、tanh和逐元素调制不包含于此，真实时间/显存必须实测。新增模块参与部署，原默认DURA权重与固定执行路径不变。

G从零单阶段30轮/7410步，B256、L189000/U1701000，原四物理项与SIG alpha1/输出梯度cap0.5，模型/CUDAprobe2026092701、shuffle2026103004、源cache01..20再01..10、AdamW0.001/warm247/cos2e−5/WD1e−4/clip1、FP32/TF32off均保持。只冻结final30，不因源结果取消完整七景；没有本次残差符号增强、教师IQ、G祖先或新场景抽样。

原F50/source4df7/defaulthead冻结，源clean实际加权SIG63000×160/40.32MB只读复用，原producer63000prefix/0.847077秒及F50/50轮11100步监督知识与成本保留，F-guided、不称F-free；旧SIG G不是祖先。新增F缓存/缓存VJP/数值复制0，在线F前缀/输入VJP各1.89m、完整F后续训练0，0TX/V/target/类损失/另训头。旧源PA、PA＋SIG、response＋SIG和其他源资格FAIL保留，不以该新结构追改旧判定。

3项聚焦检查、实际factory/config参数、aliases/初始IQ、syntheticMAC/all7入口路由通过，唯一P0/P1 PASS。launcher首步真实2非clean L_s（按variance>1e−8取前2，不读TX）检查cache/F/cap、coarse/action/prior实际梯度、conditioner连接/有限、严格E0与native/optimized原头状态；G12/fullF2/prefix4/VJP2，0模型优化，PASS立即继续。E0原prior.out和FiLM末层均零造成上游modulation梯度暂为零属预期，非零fixture已验证全参数梯度；最终学习状态再核实。

一次N607 preflight、归档SHA比较、远端编译及实际PID/CWD/argv/GPU/log增长。GPU4/6/7剩余≥12000MiB、已有compute进程≤2才加1，总≤3，每卡own1、goal≤2GPUworker；保留其他owner。固定最终训练后自动profile/source1800×7/formal168000×7，各role全7预测固定后独立CPU truth-last。主SIG47591/次pureG8cf/defaultDURA/R17对照，六景每景≥137216正确、clean≥142256、R17每景最多少1680、旧lowU70%分列；实际成本与同质量目标一起验收，不预设有效。


## 实际启动证据

VERIFIED / RUNNING，尚无性能结果。执行d32e98f97047afc78bdd89dec51f7f804358cf3a；真实2Ls smoke PASS后立即进入GPU4单worker1804637，controller1804542，已有2轮/494步。实际CWD/完整argv/commit/CUDA设备/compute PID/两份日志增长核实。准入前已有1个compute PID、剩余22883MiB，新增1个，不干预其他owner。归档SHA比较与远端编译通过。详见launch_evidence.json；固定最终30轮及全部七景自动闭环保持。
