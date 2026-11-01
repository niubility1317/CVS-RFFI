# 去除原观测代数捷径的四步IQ恢复对照

**六景0/6，合并clean未保持，clean准确率86.107143%。** cross_rx clean净变化-551，cross_both净变化-174。完整目标未达到，保持 ACTIVE 和默认 DURA。[完整性能与分组](analysis.md)。

- run_id：`20261101-phase1-Restore-StateBridge-E30-s2026092701-r01`
- group_id：`restoration-state-only-direct-iteration-single30`；类别：`cvs`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：ANALYZED（完整七景与独立复算已核实）

## 目的与对照

只删除DirectBridge估计器的原received两通道，其他初始随机状态/源配对时间/损失/30轮和四步执行匹配，自动原Main七景。

## 数据、seed与模型来源

实际数据契约、权限例外、完整seed角色、checkpoint来源和选择规则见experiment.json。
逐行配置通过config_ref/resolved_config_ref定位；待补项必须在对应生命周期补齐。

## 执行与存储

命令、环境、CWD、commit、launch owner、输出和日志路径见experiment.json。
实际PID/GPU、读取时间、remote readback、失败或替代关系在此追加，并用record命令记录证据指针。

## 结果与覆盖

原 Main168000×七景完整，六景0/6、合并clean未保持。84单元与完整分组/配对复算PASS，详见analysis.md。
源域训练完成、预测完成、评分完成及协议有效性分别陈述。不得用总索引或旧状态证明当前运行。

## 交接

记录已完成、当前run/commit、证据路径、阻塞与下一步；恢复先查原run，不重复启动。

## 单因素、源端依据与固定执行

源配对状态z=(1-t)x+t y在同时提供y/t时，t<1可代数解出x=(z-t y)/(1-t)。若第一步预测c，后续用该解析解和既定4步公式会保持c，不必学会进一步恢复。该合成IQ恒等式及固定点已验证，不读取真实源/目标/TX，也不证明实际DirectBridge一定采用捷径或最终分数无效。既有84a063c9 DirectBridge健康训练/完整Main矩阵保持，不热改、不因源端假设取消测试。[限制说明](../20261101-phase1-Restore-DirectBridge-E30-s2026092701-r01/paired_state_shortcut.md)

本独立StateBridge仅删除估计器的原received两个stem通道，输入state2+时间1+坐标1，4676898参数。工厂仍从零seed2026092701，保留相同随机初始化的其他tensor及stem的state/time/coordinate通道，未加载任何旧G权重。连续160/四feature与time FiLM、单位clean预测、IQ+.05STFT+.1输出F指导+.1latent指导+.1rawRMS约束、全局clip1及全部优化预算匹配。估计器不读取兼容接口中的received_unused；固定state/time改变该参数100倍时输出逐值一样。received只初始化z，四次共享权重依次用remaining4/3/2/1执行z_next=z+(predictedclean-z)/remaining，末步直接预测clean。无部署clean、F0、SNR真值、额外观测、随机或跨query统计。

原Ls6300/Us56700、R07 20draw+前10复用、order2026110102、B256tail24/247steps、AdamW/.001-warm247-cos2e-5/WD1e-4、scratch单阶段30/7410均保持。每条训练1core，用合法配对构造state=(1-t)detachedclean+t detachedreceived，t1概率1/2、.75/.5/.25各1/6/globalCUDAseed2026110103，源clean进入训练state的事实明确披露；各time计数与指标单列，不能用较容易中间输入的均值声称恢复性能提高。teacher F0源clean连续160/f57尺度与RMS只进loss，无TX/logits/CE/六类原型/新身份头，无V/OE/novel/目标训练或选模，OE仅冻结诊断，F-guided而非F-free。无旧G/coarse/DURA/神经IQ教师祖先。

12项重点检查（新5+Direct7）及一次独立P0P1通过。真实源2训练梯度、完整4步E0与严格重载为launcher首步，PASS立即30；E01..30只技术恢复/final30固定，源质量不取消Main。每版按既定原Main126000+42000各七景全部预测固定后独立CPU truth-last/原共享锁，raw/DURA同IQ-ID精确复用，新G全4步推理；逐景至少140345正确=83.538690%，距原地面TriSIGTIME SSL/defaulthead raw clean86.538690%≤3pp并保持两个总体clean，完整TX/RX/day/净纠错及相对clean变化报告。对照的目标分数不回流、单seed与已曝光Main限制保持，默认DURA不变。

静态4×260633600=1042534400实数Conv/Linear/attention MAC/包、0FFT/CG，训练每记录1core/部署4core后F0一次；排除F0/norm/activation/softmax/FiLM逐点/插值/投影/bridge更新/内存，不是实测时延。当前两个独立run各最多1条NN，本任务合计最多2条；GPU4/6实时free>=12000MiB/原compute<=2/add1总<=3，其他owner/健康Direct计划不改。环境仍N607 CVS-RFFI Torch2.1+cu121/FP32-TF32off。独占run-root/精确命令/停止规则/产物见登记。


## 仅状态时间四步恢复器已运行（2026-11-01）

`20261101-phase1-Restore-StateBridge-E30-s2026092701-r01`，实际代码`c50a9159b1fb891c6f1a6f8ad4bfa2d7eb66222e`，4676898参数，从零单阶段30/7410、训练每记录1次estimate、部署4次共享权重，received仅初始化state，估计器不读取原received条件，无额外观测或部署clean/F0。源IQ配对构造t1/.75/.5/.25状态，原F0连续160与rawRMS参考loss-only；0TX/V/OE/目标训练或选模，OE仅冻结诊断。结构/训练状态/执行共同变化的系统比较，不冒称单因素。

VERIFIED / RUNNING：12重点相关检查/一次P0P1、真实源2训练梯度与完整四步E0/严格重载PASS。首轮247更新耗时78.104723秒，E01模型/AdamW/CPU-CUDA RNG独立CPU读回PASS，E0与固定seed factory逐张量一致；4个time实际计数和各项指标重组PASS，总loss确认原权重各加一次。GPU6准入free20513MiB/原2compute/仅新增1条，PID/CWD/命令/CUDA/GPU和日志核实，其他owner保持。固定final30后自动原Main168000×七景、全部预测后独立truth-last，当前性能PENDING；主验收仍六景各距raw clean≤3pp并保持clean，性能主线、效率支线和默认DURA不变。

## StateBridge完整七景完成

VERIFIED / ANALYZED。全G从零单阶段30/7410，部署固定4次共享权重，两个Main总体各七景全部预测后独立truth-last；4阶段exit0/liveempty。84单元/2604组/1736raw配对/1302额外配对复算PASS，误差1.24e-14、raw/DURA判决差0。完整60轮/stdout/源顺序draw/固定时间协议/从零E0/总loss与每t重组/30快照E30-final核实。完整目标未达到，保持 ACTIVE 和默认 DURA。
