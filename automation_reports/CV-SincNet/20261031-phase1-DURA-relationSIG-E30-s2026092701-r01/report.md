# 跨接收机签名关系指导：恢复器从零单阶段30轮

- run_id：`20261031-phase1-DURA-relationSIG-E30-s2026092701-r01`
- group_id：`phase1-dura-cross-RX-relation-SIG-E30`；类别：`cvs`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：ANALYZED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

原G266951/物理4项/E30预算不变，源F真实SIG缓存与源RX构成cross-RX余弦距离关系，单prefix/VJP及逐IQcap0.5；无新推理模块，完整七景。

## 数据、seed与模型来源

实际数据契约、权限例外、完整seed角色、checkpoint来源和选择规则见experiment.json。
逐行配置通过config_ref/resolved_config_ref定位；待补项必须在对应生命周期补齐。

## 执行与存储

命令、环境、CWD、commit、launch owner、输出和日志路径见experiment.json。
实际PID/GPU、读取时间、remote readback、失败或替代关系在此追加，并用record命令记录证据指针。

## 结果与覆盖

完整结果见analysis.md/results.json，分组和成本保留。
源域训练完成、预测完成、评分完成及协议有效性分别陈述。不得用总索引或旧状态证明当前运行。

## 交接

记录已完成、当前run/commit、证据路径、阻塞与下一步；恢复先查原run，不重复启动。


## 固定跨RX签名关系对照

G从零单阶段30轮/7410，原266951参数、物理4项、初始化/RNG/LR/数据/source01..20再01..10和推理路径保持。仅将指导由逐样本SIG相对MSE改为跨源RX的余弦距离关系，实例SIG误差仍记录但不进入损失。所有已完成源FAIL、旧正式输出与默认DURA保持；不据本次源指标取消完整七景。

F为原F50/source4df7/defaulthead，源clean实际weighted SIG160缓存63000×160/40.32MB只读复用。每batch附原source RX列作为训练损失上下文，161列不进入G。d=1−normalizedSIG cosine；仅RX不同的所有pair，e_i=mean_pair(d_restored−d_clean)² / mean_pair(d_clean²).clamp_min(1e−4)，无跨RX边时e=0并记录active。对所有e_i的SUM求一次F input VJP，包含pair两端，再按每条IQ的0.5×最终IQ NMSE梯度范数上限detached注入。该局部预条件梯度不冒充加权全局pair风险的导数，surrogate数值用于损失记账；最外层batch mean只除一次B。

关系损失允许共同特征旋转，不能凭关系保持保证原头预测或真实TX，因此物理IQ锚点和全部真实七景验收必需。方法受[Park等，CVPR2019关系蒸馏](https://openaccess.thecvf.com/content_CVPR_2019/html/Park_Relational_Knowledge_Distillation_CVPR_2019_paper.html)思路启发，是本项目自定义IQ余弦关系目标，不当作原RKD复现或已有RFFI效果证明。teacher特征可能仍含RX/信道/波形信息，不预先声称硬件分离或RX不变。

原G factory、alias和所有初始化参数保持，无新推理模块、参数、RX输入或查询间统计。新增计算为source pair矩阵及临时target打包，没有额外F forward/VJP；训练online prefix/VJP各1.89m、fullF后续0，新持久F缓存/完整target数值副本0。临时batch复制和矩阵计算计入实际时间。原F50/11100源监督知识与原缓存63000prefix/0.847077秒成本保留，F-guided，不称F-free；旧SIG G不是祖先。0直接TX/V/target/类损失/另训头。

3项聚焦检查、实际factory/config/单E30-all7入口、有效新训练hook/161列targetwrapper、物理4权重/实例误差仅诊断/RNG隔离通过，唯一P0/P1 PASS。launcher首步真实2非clean L_s RX1/3检查实际F/cache、两端active/非零cap、三G模块梯度、严格E0/nativeoptimized/F原头state，G8/fullF4/prefix2/VJP2，PASS立即继续。每row最终30后自动同卡profile、sourceV1800×7/formal168000×7，各role全七景固定后独立CPU truth-last。

单seed2026092701、AdamW0.001/warm247/cos2e−5/WD1e−4/clip1/FP32/TF32off，model-CUDAprobe2026092701/NumPy shuffle2026103004，所有pair确定性，无新随机流。GPU4/6/7/free≥12000MiB/existing≤2加1总≤3、每卡own1/goal≤2GPUworker/ownCPUtruthscorer≤1，其他owner保持；一次preflight/单归档SHA/compile/实际PID-CWD-argv-GPU-loggrowth。主SIG47591、次pureG8cf/defaultDURA/R17：六景各≥137216正确、clean≥142256、R17每景最多少1680及旧lowU70%分列。完整目标ACTIVE，已有曝光工程基准不称新盲测。


## 实际启动证据

VERIFIED / RUNNING，尚无性能结果。执行53aec99292cde105ab6b582ac8fd56df1e71e798；真实2非clean L_s RX1/3 smoke PASS后进入GPU4 worker1865565/controller1865475，已核实3轮/741步、两日志增长。实际CWD/完整argv/commit/CUDA/compute PID核对，准入前0个compute PID/剩余24124MiB，单任务新增1个。归档SHA比较与远端编译通过；固定最终30轮/profile/全部七景自动truth-last保持。详见launch_evidence.json。


## 完整训练与正式结果

VERIFIED / ANALYZED。跨源RX余弦关系SIG指导G从零单阶段30轮/7410步，L189000/U1701000，无G祖先；完整G266951参数、推理路径和物理4损失保持。只改变F指导，源RX仅用于batch训练损失、不进入G。SUM关系VJP含两端，逐record IQ梯度cap0.5后detached注入，是局部预条件代理导数；实例SIG误差只记录。源SIG40.32MB只读复用、逐batch161列临时上下文，新完整数组复制/缓存F0；历史63000prefix/0.847077秒与F50/11100源监督知识成本保留，F-guided。训练prefix/VJP各1.89m、完整F后续0，F不变，0直接TX/V/target/classloss/新头。每role全七景固定后truth-last，126正式/105源V/252双对照配对分组/14IQ独立复算。全30轮loss/曝光/视图/cap核对，按实际shuffle独立重放跨RX有向配对数与active记录数一致。六星地较SIG提高0/6、纯G提高3/6、默认DURA提高0/6。clean 84.842857%，低城区 68.009524%；低城区较SIG -0.825595、纯G +0.636905、默认DURA -0.880357个百分点。六景3pp通过0/6，clean严格保持=True，R17逐景1pp=False，旧低城区70%=False。G训练及保存1026.200308秒、含worker准备1027.364699秒、控制器至自动分析1196.162898秒。ConvLinearMAC25068072/2prior4CG27FFT，同卡G+F B1默认4.592008ms、本版4.585469ms。共享GPU单次墙钟不证明因果加速，特征关系不保证纯TX/硬件不变性，有限模型成绩不认证物理恢复极限。默认DURA保持，完整目标ACTIVE。

详见[完整分析](analysis.md)、[训练与七景图](training_and_formal.png)、results.json和transfer_diagnostics.json。controller1865475/worker1865565自然结束，无目标反馈或选择性重跑。
