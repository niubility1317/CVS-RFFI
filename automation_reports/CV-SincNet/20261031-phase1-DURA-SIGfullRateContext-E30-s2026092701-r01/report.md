# 预测特征引导的全采样率IQ残差分支：从零单阶段30轮

- run_id：`20261031-phase1-DURA-SIGfullRateContext-E30-s2026092701-r01`
- group_id：`phase1-DURA-full-rate-IQ-residual-pair-E30`；类别：`cvs`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：ANALYZED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

保留原main344295工厂及全部损失，两个prior各新增全率9→12卷积、三膨胀深度卷积残差与零输出映射；由同prior预测SIG经零初始化160→72映射调制三个新块。

## 数据、seed与模型来源

实际数据契约、权限例外、完整seed角色、checkpoint来源和选择规则见experiment.json。
逐行配置通过config_ref/resolved_config_ref定位；待补项必须在对应生命周期补齐。

## 执行与存储

命令、环境、CWD、commit、launch owner、输出和日志路径见experiment.json。
实际PID/GPU、读取时间、remote readback、失败或替代关系在此追加，并用record命令记录证据指针。

## 结果与覆盖

完整结果见analysis.md/results.json，保留全部分组和成本。
源域训练完成、预测完成、评分完成及协议有效性分别陈述。不得用总索引或旧状态证明当前运行。

## 交接

记录已完成、当前run/commit、证据路径、阻塞与下一步；恢复先查原run，不重复启动。


## 固定假设、方法与测试矩阵

预测特征引导的全采样率IQ残差分支。源诊断已排除两条预设解释：深层预测器在同源接收机的新物理记录上仍能改善SIG及head方向误差，但恢复输出没有同步获益。本次单独测试增加全采样率IQ输出分支，不声明已证明某一解码模块或高频损失是因果瓶颈；原UNet已有全率skip。

原main344295随机架构、浅中层SIG预测和原三层条件解码保持。每个prior新增同9通道输入→12通道k3/GN3/SiLU、三个DWk3膨胀2/4/8＋PWk1＋GN3/SiLU残差、12→2零初始化输出。新增delta与原delta相加后仍由原SCALE缩放，再进入原CG。无引导版的新分支不使用SIG；引导版使用本条预测SIG的160→72零初始化映射，在三个块内施加bound0.1的scale/shift。两版共有新分支参数初始化相同；旧参数/SIG/IQ/CPU-CUDA随机流保持，G祖先0。没有新增求解次数、第二物理观测或部署标签输入。

完整G参数369547，新增25252，Conv/Linear MAC 25622568已由实际完整前向hook核对。两prior/四CG/27 FFT-IFFT不改；新增GN/激活的成本以实测计入，不以MAC或小网络推断更快。原四物理＋输出SIG alpha1/IQ梯度cap0.5＋0.01×两prior相对SIG误差均值不改，辅助梯度在输出cap外、全G clip1。F-only源SIG40.32MB只读复用，无新缓存F，原F50训练50轮11100步和缓存63000prefix/0.847077秒知识成本披露；F-guided，不能称F-free。

两版均从零单阶段30轮/7410步，B256、L189000/U1701000，原缓存01..20后01..10；native训练和冻结路径、损失、优化器、随机顺序均匹配。固定final30；各版自动profile、源1800×7和正式168000×7，每个角色七景预测全部固定后独立truth-last。即便源分数差也完成正式测试，不从目标结果选模、调权重或选择性重跑。主对照原main4452，两新版互比并列原SIG/pureG/default/R17。六景逐个≥137216正确、clean≥142256、R17逐景损失≤1680，low urban70%另列。

5项聚焦测试及两种真实CLI/全factory/参数/实测MAC/别名/原main初始IQ与随机流/共有新分支初值/实际objective/新梯度/targetdetach/路由/独立输出/共同CPU锁/七景检查PASS。每候选唯一独立P0/P1均PASS，无P0/P1。发布首步真实2条非clean L_s smoke：G12/完整F4/prefix2/VJP2，检查原cap、F冻结、真实新输出非零梯度、零输出下内部初始零梯度、严格E0重载和native/optimized一致、新分支两次及六块B×12×256，然后立即训练。

GPU4/6/7中启动前显存余量≥12GB、既有compute≤2时单卡新增1个，总≤3；每卡本任务1个、整个目标≤2GPUworker。CPUtruth共用plain新run的score.lock，输出独立；不得修改其他owner。一个不可覆盖root、一个代码commit、一次archive SHA/远端编译/预检及启动后PID-CWD-命令-GPU-日志增长核实。默认DURA保持，完整目标ACTIVE。


额外聚焦权限反例与资源边界检查2项PASS：错误教师/源合同/训练标签权限/目标训练/F更新或schema在建模前拒绝；新增进程计入总容量并保持显存门槛。


## 实际启动证据

VERIFIED / RUNNING，尚无正式性能结果。执行43577dc71dfdab6382dcf94e75c3b699b2c5122e，controller 2054413、GPU 7 worker 2054717。真实2条非clean L_s smoke PASS：原main初值IQ、原SIG cap、三个G模块与新增输出实际非零梯度、初始内部零梯度、严格E0/nativeoptimized/F冻结、新分支两次及六块B×12×256均核实。已完成1轮/247步；实际CWD/argv/commit/CUDA/compute PID与日志增长通过。准入前compute 1个、剩余22883MiB，新增1个。归档SHA、远端编译和预检通过；单阶段30轮与自动七景保持，CPU共同锁、独立输出，默认DURA不变。


## 完整训练与正式结果

VERIFIED / ANALYZED。全率特征引导版，从零单阶段30轮/7410步，L189000/U1701000，G祖先0。保留原main344295随机架构及SIG预测/原条件解码，每prior新增全率9→12 k3/GN3/SiLU、三DW膨胀2/4/8-PW-GN-SiLU残差与12→2零输出；新增delta在原SCALE内相加。无引导分支不使用SIG，特征引导分支使用同prior预测SIG的160→72零初始化映射/三块bound0.1调制。共有新分支初值匹配，旧初值IQ/SIG/RNG保持；真实新输出有非零梯度，初始内部零梯度符合零输出设计。完整G369547参数/ConvLinear MAC 25622568/两prior四CG27FFT，native全部路径，原四物理+输出SIGalpha1/cap0.5+0.01mean2priorSIG辅助保持。F-only缓存40.32MB只读复用/newFcache0，F50/source4df7/50轮11100步及历史缓存成本披露，F-guided；在线prefix/VJP各1.89m、完整F训练0、0TX/V/target训练或新分类头。源泛化f19的两条解释FAIL只支持独立输出路径假设，不证明某一解码模块因果失效；原UNet已有全率skip。固定final30，各角色七景预测后独立truth-last，126正式/105源V/378配对分组/14IQ独立复算，30轮loss/曝光/视图/cap/来源及配置核验。六星地较原条件版提高0/6、较另一全率版提高0/6、较原输出SIG提高0/6、较默认DURA提高0/6。clean 84.758929%、low urban 68.304167%，3pp通过0/6、clean严格保持=True、R17逐景1pp容忍=False、旧lowU70%=False。G训练保存1246.039412秒、含准备1247.141547秒、控制器至自动分析1439.785878秒；同卡G+F B1默认5.067270ms、本版5.305742ms。共享负载墙钟不作因果提速或同质量速度比；波形和固定F可识别性分别判断。默认DURA保持，完整目标继续逐项验收。

详见[完整分析](analysis.md)、[训练七景图](training_and_formal.png)、results.json与transfer_diagnostics.json；控制器和worker自然退出，产物独立读回。
