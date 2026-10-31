# 精确非线性F特征指导：从零单阶段30轮

- run_id：`20261031-phase1-DURA-exactReference-E30-s2026092701-r01`
- group_id：`phase1-dura-source-exact-reference-E30`；类别：`cvs`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：ANALYZED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

同266951G/物理四项/源视图/E30预算，固定source10软跨RX目标，在线冻结F精确输入梯度/每步cap1/一次F reverse注入；新目标缓存2.52MB、0J读取。全七景。

## 数据、seed与模型来源

实际数据契约、权限例外、完整seed角色、checkpoint来源和选择规则见experiment.json。
逐行配置通过config_ref/resolved_config_ref定位；待补项必须在对应生命周期补齐。

## 执行与存储

命令、环境、CWD、commit、launch owner、输出和日志路径见experiment.json。
实际PID/GPU、读取时间、remote readback、失败或替代关系在此追加，并用record命令记录证据指针。

## 结果与覆盖

完整结果见下方、analysis.md与results.json；保留每row与RX/day/TX/scene/K/seed的对应关系。
源域训练完成、预测完成、评分完成及协议有效性分别陈述。不得用总索引或旧状态证明当前运行。

## 交接

记录已完成、当前run/commit、证据路径、阻塞与下一步；恢复先查原run，不重复启动。


## 固定精确F指导30轮

原266951参数G从零单阶段联合30轮/7410步，物理四项、seed/CUDAprobe、随机批次、学习率、源01..20后01..10保持。将head5缓存近似换为同源10维精确F约束，同时改变约束维数和导数方式，不能把结果归因为单项因素。beta0.25、alpha0.0033130025035573727、IQ梯度cap1保持；guide已含alpha，物理loss直接加guide。F50/source4df7/原默认头冻结、optimizer只G、无直接TX/V/target。

新特征目标63000×10 FP32/2.52MB由已有source F160/软RX均值/合法基底生成，生产F0/VJP0，不读J或GN修改IQ。训练在线F1890000输入+1890000输入反向，一次reverse后注入detach限幅梯度，F50历史知识/成本与新增缓存/真实时耗全部披露，不称F-free。F加载及额外F路径隔离Torch/CUDA随机流，避免扰动G/probe；固定最后30轮。

4项新集成测试、实际入口、唯一P0P1通过；真实2Ls三个G模块/精确IQcap/F冻结原头/严格epoch0重载后立即继续。满30自动profile、source1800×7和target168000×7，每role全预测固定后独立truth-last，即使源成绩差仍完成新G正式矩阵。每六星地≥137216正确且clean≥142256，R17每景1pp及旧低城区70%另列。默认DURA与完整目标保持。


## 实际启动证据

VERIFIED / RUNNING。执行cc6440f3be59239e54cf4e6c900d75d3c8314ce8已提交推送并独立核对远端OID；一次N607 preflight/单归档传输SHA/remotecompile通过。实际controller 1570379/worker 1570547/GPU 4 的CWD、完整命令、CUDA、GPU和日志增长核实，旧own句柄全部消失。启动前同卡1个计算进程、free 22883MiB，新增本任务1个，其他owner保持。真实2Ls精确IQ梯度cap/三个G模块梯度/原F默认头冻结/严格epoch0重载通过；F加载后CPU/CUDA RNG恢复已实际核实，F路径也隔离随机流。至少完成2轮，完整前缀预算/加权loss/源视图/Fforward-VJP-targetrefs/逐批次cap核对通过，已验证F输入126000/输入VJP126000，最大压力1.000000119。

新增source target cache63000×10 FP32/2.52MB，生产0.509850秒；独立复算630000分量、basis10与512资格记录，最大差0.0。生产新F/VJP0，不读J或GN修改IQ；训练在线F完整预算1890000前向+1890000输入反向，原F50及缓存/源资格成本保留，是F-guided。当前正式结果待产出；固定最终30轮后自动profile/source1800×7/target168000×7/全部预测后独立truth-last。默认DURA保持，目标ACTIVE。


## 完整训练与正式结果

VERIFIED / ANALYZED。G从零单阶段30轮/7410步，L189000/U1701000；完整30轮loss/曝光/缓存视图/逐批次压力检查通过。source10跨RX/day软参考目标2.52MB/0.509850秒，生产新F/VJP0、训练在线F1890000前向+1890000输入VJP，训练不读旧J；F最终状态不变、原默认头保持。继承F50监督知识，属于F-guided；约束5→10维与导数方式同时改变，不作单因素因果归因。七景×168000全部预测固定后truth-last，126正式、105源V与252双对照配对分组独立复算，14个源/正式IQ场景误差重新核对。clean 84.970238%，low urban 68.219048%；低城区较缓存5维 -0.436310、纯G +0.846429、默认DURA -0.670833 个百分点。六景3pp通过0/6，clean严格保持=True，R17逐景1pp=False，旧低城区70%=False。G训练1261.405657秒，含worker源准备1262.390721秒，controller至全量分析1441.549878秒。266951参数/25.068072m ConvLinear MAC/2prior4CG27FFT，默认DURA保持；完整目标仍ACTIVE。

详见[完整分析](analysis.md)、[完整曲线与正式结果](training_and_formal.png)、results.json与transfer_diagnostics.json。controller/worker已自然结束。源可达性和局部cost资格通过不替代G全量真实身份测试；本版结果不作为理论恢复上限，也未用于回调beta/alpha、选epoch或选择性重测。
