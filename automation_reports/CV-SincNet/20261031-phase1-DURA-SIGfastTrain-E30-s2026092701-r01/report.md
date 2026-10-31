# 原特征指导的训练期插值执行优化：单阶段30轮

- run_id：`20261031-phase1-DURA-SIGfastTrain-E30-s2026092701-r01`
- group_id：`phase1-dura-original-SIG-training-only-fixedgrid-E30`；类别：`cvs`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：RUNNING（实际状态按events.jsonl及独立证据更新）

## 目的与对照

完整原SIG训练目标不变，仅训练G的四Core采用已有固定网格插值；训练结束恢复原生factory，冻结预测为原native路径。

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


## 固定训练执行对照

完整原SIG-only配方保持：G266951从零单阶段30轮/7410，原四物理项与actual SIG alpha1/IQ梯度cap0.5、L189000/U1701000、source01..20后01..10、model/CUDAprobe2026092701、NumPyshuffle2026103004、AdamW.001/warm247/cos2e−5/WD1e−4/clip1/FP32/TF32off。与独立coarseAnchor对照分开，本版不含其新增项，也不增加新目标、增强、教师IQ或网络参数。

唯一差异是训练构造G时，在已有四个Core的12/15次固定二倍上采样使用既有半像素slice实现，action另3次native。try/finally在训练结束或异常时还原原生factory；保存state_dict后，冻结加载每次断言四个Core均为原native类，profile和源/正式预测均走原native执行。初始化参数、共享关系、随机流和部署结构保持；局部数值容差一致不意味着7410步AdamW轨迹逐位相同。

依据为此前已完成源零更新插值诊断的完整前向/损失/全部裁剪后梯度rtol2e−5/atol2e−6检查，以及同卡B256三轮交替各3次：native221.756580ms、slice189.741313ms、中位减少14.437%，三轮正收益。该既有匹配步骤证据复用，不重复CUPTI或放大为当前SIG/E30/整轮提速承诺；新run仍完整报告实际训练、全流程、质量与时延。

3项聚焦检查覆盖工厂成功/训练异常/安装异常均还原。actual G266951、原训练objective身份、真实四Core训练slice→native还原、相同初值与RNG、冻结native边界、单E30七景入口/help通过。唯一独立P0/P1及仅评分互斥问题的定点修复PASS。首步真实2非clean L_s比较native/slice IQ、损失、全部裁剪后G梯度容差，检查原SIGcap/三G模块梯度、严格E0 state重载原native Core、nativeoptimized/F冻结；G14/fullF4/prefix4/VJP4，PASS直接训练。

源F-only SIG63000×160/40.32MB只读复用，新缓存F0；原F50/source4df7/50轮11100步及缓存63000prefix/0.847077秒知识成本披露，F-guided。在线prefix/VJP各1.89m、完整F训练0，G祖先0，0直接TX/V/target/classloss/另训头/sourceRX输入。固定final30，自动native profile/source1800×7/正式168000×7，各role全七景预测完成后独立CPU truth-last，源差不取消测试，不按目标结果选模或重跑。

GPU4/6/7/free≥12000MiB/existing≤2新增1总≤3、own1/card、goal≤2GPUworker。新CPU评分与当前coarseAnchor旧worker锁定同一既有run score.lock，保证own CPUscorer≤1，各自输出根完全独立；旧健康进程无需修改。其他owner保持。主原SIG47591，次pureG8cf/defaultDURA/R17；六景各≥137216正确、clean≥142256、R17逐景最多少1680，旧lowU70%单列。126正式/105源/252双对照/14IQ和全30轮loss/曝光/视图/cap均复核，默认与完整目标保持。


## 实际启动证据

VERIFIED / RUNNING，尚无性能结果。执行84d9d483ab2aad2ee4e433949d11f09238dbd6c5；真实2非clean L_s native/slice loss与全部裁剪梯度、初始化/原SIGcap/严格重载native/Core和F冻结检查PASS后进入GPU7 worker1896763/controller1896664。已核实1轮/247步、实际CWD/完整argv/commit/CUDA/compute PID和日志增长，准入前1个compute PID/剩余22883MiB，新增1个。归档SHA比较与远端编译通过；固定final30、原native profile/全部七景自动truth-last，CPU scorer共用当前coarseAnchor既有锁。
