# 完整源域 F＋跨 RX：E30 practical 训练边界混合预登记

状态：RUNNING，已独立核实启动；尚无本行正式结果。代码/config commit：8cf7ceed44f9818390332cebd9f4c34c3b8b7967，GitHub 分支 OID 已独立核对一致。唯一新行、seed与路径在experiment.json；本地4聚焦测试与2包真实模型合成前向PASS，未读取真实IQ/checkpoint/目标。

源依据：复用已完成全源V1800、两信道draw的冻结E20诊断，reflect→edge在固定接收状态/AGC下保持195个内部点不变，却使B01 mid/low源准确率降低10.833333/9.25pp。这只证明边界敏感性，不证明reflect错误、edge更真实或训练能恢复。当前packet只有256点，不能臆造真实前史。

唯一假设：在保持完整practical传播、噪声及接收补偿依赖时，训练中交替两种合成前史可减少人工前缀依赖。新行BCTX每轮222批，奇数reflect/偶数edge，各111批；原生接收机按该边界完成处理并重算AGC，不使用冻结AGC控制图作为训练数据。所有其他Config字段不变；边界不依赖TX。

固定F＋跨RX同TX：三SIG_TIME、CENTER_25K/CENTER_UNITS、NoExplicitCFO、joint160、原默认cos30、900722参数/206状态，骨干与头联合更新。原L6300/U56700隐藏/V27000、全部RX1/3/4/6/8、day1/2/3、6TX、split392005及7/63/30不变，无额外Ldev/留RX，训练不建目标loader。

唯一新模型scratchE30＝20L＋10U、6660更新；L每物理包ordinary/practical(ordinary)两半各0.5归一化CE/KL，同IQ两次RDrop、4L；原clean半SupCon、原U/ownEMA/旧门控不变。训练三场景原nominal均匀，几何/遮挡/传播/噪声/质量/相位/RXIQ补偿依赖均保留。EQoff/G1/referenceSNR30/fs25MHz/fc2.45GHz，无额外As后RMS，不反演TX。

与已完成同预算B01普通reflect拼接主比较，B00无As作clean成本参照；两者只读复用，不作初始化。当前只检验A_s组织，不称Ag、四格交互或联合突破。末轮固定FINAL30后不看分数选模，自动原clean＋六practical全部168000物理样本预测，再独立exec truth-last评分；原正式IQ/SNR/reflect及困难样本不变。

新增预算30轮/6660更新，父训练0；预计L网络3409920包、原U loader567000、各边界426240卫星包。student/teacher/V/compatibilitysmoke/正式预测/墙钟按实际单列。历史全源B00 E20/4440只供launcher首步2合成包strictload noquery smoke后丢弃，不继承E50/E200或任何父状态。

现有探索目标用于终局描述：相对B01 urban+3pp、low+5pp，保护同预算B00 clean，low>60%、suburban距自身clean≤2pp；不是结果保证、目标反馈晋级或重跑规则。负结果保留，不调边界比例、seed或预算救跑。若完整源训练和默认头最终识别没有改善，则不宣称该假设有效。

真实本地检查：4项测试，8配置/6call/4namespace/7物理负例；48合成native calls证明noeq噪声、接收状态/补偿及preAGC内部195点匹配，输出仍按各自nativeAGC计算。另2合成NN证明与原E30 B01初态206项相同、参数900722、默认头/Sinc/三SIG_TIME梯度健康；0真实IQ/CP/更新。compile PASS。第一次直接-m导入缺少已拆分legacy依赖，使用原验证过的本地装配后PASS；已知Torch/NumPy导入警告保留。

实际本地命令：

```text
E:/type10-7/local_envs/ssr-gpu/python.exe -X utf8 E:/type10-7/local_artifacts/cvs_FcrossRX_fullsource_boundarymix_E30_20261017/local_checks.py
E:/type10-7/local_envs/ssr-gpu/python.exe -X utf8 E:/type10-7/local_artifacts/cvs_FcrossRX_fullsource_boundarymix_E30_20261017/local_model_checks.py
```

普通N607/CVS-RFFI/FP32/CPU2，训练GPU在原共享lock下按每GPU≤3个总实验进程（含其他owner）、free≥12000MB选择；不修改健康任务。一个launch owner/root，一个不可覆盖run及release，commit是代码/config身份；一次preflight、一个归档与一次传输SHA比较、remotecompile、首步realCPsmoke、启动后PID/CWD/argv/GPU/log增长。

训练CWD：/home/szu2070436088/2510044040/CV-SincNet/releases/cvs_FcrossRX_fullsource_boundarymix_E30_20261017_r01。实际计划命令：

```text
/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -u /home/szu2070436088/2510044040/CV-SincNet/releases/cvs_FcrossRX_fullsource_boundarymix_E30_20261017_r01/experiments/cvs_fullsource_boundary26/runner.py
```

run：/home/szu2070436088/2510044040/CV-SincNet/runs/20261017-phase1-FcrossRX-fullsource-boundarymix-e30-s2026092701-m1-r01；本地：E:/type10-7/local_artifacts/cvs_FcrossRX_fullsource_boundarymix_E30_20261017。Config/actualnativeargs/模块__file__/两边界firstbatch/完整physics及L-U序列/6660步/30轮曲线/默认头及Sinc移动/NN/最终CP和freeze/七预测与独立评分/终局原B00-B01比较均保存。IQ/CP/原始log/生成运行输出不入Git。

技术停止：本run角色、来源、CWD、输出冲突、非有限或跳步、未6660、边界越过L训练scope、预测或truth隔离错误即保留产物，不自动重试，不影响其他owner。科学低分不技术停止。恢复先核实原进程与输出，不覆盖或重复launch。独立 P0/P1 审查 PASS，无直接 P0/P1 缺陷；沿现有授权发布。

此前谱响应/PSD正仿射/局部谱峰与urbanSSL完整负结果已复核，本候选不以改RX或门槛重跑这些路线。新Ag、B10/B11、低urban60%及跨seed联合收益仍未证明；目标保持ACTIVE。


发布读回：普通账户 N607 preflight PASS；新 run/release 无冲突；一个归档传输 SHA 比较 VERIFIED、远端 compile PASS；launcher 首步真实同契约 E20 checkpoint strict-load no-query smoke PASS（2 合成包，随后丢弃）。dispatcher 447326，GPU 0 worker 447354，真实 CWD/argv/run-root 已匹配，队列一行 active、0 failures。启动及 smoke 均不是本行训练/测试完成。


终局分析准备：analysis/cvs_boundary_E30_results_20261017.py 复用既有全30轮分析，等 B00/B01/BCTX 三行全部七视图预测、独立评分和E30/6660完成证据后才读取truth。输出三组配对修复/新增错误、固定B00 clean正确集合保持率、默认头margin/30、最差RX/F1、全部训练曲线、原生物理统计与180个TX×RX×场景×边界覆盖单元；无IQ/checkpoint读取、无NN/拟合。7个聚焦测试PASS，含新行未完成时不得先读truth或创建分析输出。正式样本未保存逐包状态/质量标签与门控激活，不能编造这些分层准确率。


冻结后源域机制计划（本行尚未终局评分前固定）：独立 CPU2 程序 analysis/cvs_boundary_E30_source_20261017.py，等待 B00/B01/BCTX 的 FINAL30/6660 及全部七视图完成元数据后，先 strict-load 三个真实 E30 默认头并各做 2 合成包 smoke，再读取原 V 面板。复用 P24 的同一 V1800、90 单元各20包、2 信道 draw、3 场景的 full 与 edge_boundary 成对观察；不重抽信道。edge 保持 full 接收状态/补偿及 AGC，属于受控边界效应，不能当作重算接收机后的总效应。源缓存实际载入 V27000 和 L6300（L不消费）、U0；保持原角色，无新增留域。

本机制计划每模型23402NN（clean1800＋两draw×三场景×两条件×1800＋smoke2），三模型70206NN；新增训练/头拟合/teacher/物理信道调用均0。逐项检查冻结state_dict不变，保存原time/frequency/PA/reference/signature分支、真实静态融合增益、固定B00源clean正确集合保持率、accuracy/F1/margin30及原full状态/锁定/接收新增噪声比分层。该质量比不是总SNR，不读取目标IQ、成绩或truth。只有一个模型seed，两个draw不算跨seed。

源解释条件：仅当两个urban×两个draw的edge−full准确率差，相对B01均改善至少3个百分点，且各cell的full准确率下降均不超过0.5个百分点，才支持边界敏感性下降；它不是目标访问、扩seed、选模、参数修改或重跑门槛。正式成绩仍由原自动全七环境评分产生。本地3项聚焦测试PASS，含13个原角色/受控条件负例以及BCTX未完成前禁止打开源数据/CP或建立输出；compile PASS。当前该程序尚未执行，无新机制结果。

补充尺度核验：实际默认头 ResidualPhysicalHead 的 scale 固定30；160维 gain 是 identity＋tanh(gain)×physical 的特征融合系数，后续joint及weight均L2归一化。分析margin/30口径成立，不能将gain的训练移动当作logit尺度提高。核验只构造默认模型读类定义，0NN/IQ/CP/更新。


源结果独立读回：analysis/cvs_boundary_E30_source_readout_20261017.py 在 SOURCE_BOUNDARY_MECHANISM_COMPLETE 后，用单独的混淆矩阵累加实现复核全部源分层、clean指标、修复/破坏和固定B00 clean正确集合保持率，再按原定四个urban×draw单元解释条件读数。计划输出source边界、质量、状态和现有静态增益四张PNG/PDF，以及90个现有分支/场景/条件汇总。无新IQ/CP读取、NN、更新、目标成绩或truth读取。4个聚焦测试PASS（手算指标、全部单元解释条件及full保持、完成状态和权限负例、质量区间精确边界）；目前没有真实读数或图表，不能把计划路径当作已生成产物。
