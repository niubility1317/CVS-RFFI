# 源残差符号多样性：原时序签名恢复器从零单阶段30轮

- run_id：`20261031-phase1-DURA-antitheticNoise-E30-s2026092701-r01`
- group_id：`phase1-dura-source-noise-diversity-E30`；类别：`cvs`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

保持SIG-only四物理项/原G266951/固定E30，独立Bernoulli.5翻转源噪声残差并按原接收RMS归一化，同步缩放源Action/variance；复用clean SIG缓存，完整七景。

## 数据、seed与模型来源

实际数据契约、权限例外、完整seed角色、checkpoint来源和选择规则见experiment.json。
逐行配置通过config_ref/resolved_config_ref定位；待补项必须在对应生命周期补齐。

## 执行与存储

命令、环境、CWD、commit、launch owner、输出和日志路径见experiment.json。
实际PID/GPU、读取时间、remote readback、失败或替代关系在此追加，并用record命令记录证据指针。

## 结果与覆盖

完整结果见analysis.md/results.json，完整分组和成本保留。
源域训练完成、预测完成、评分完成及协议有效性分别陈述。不得用总索引或旧状态证明当前运行。

## 交接

记录已完成、当前run/commit、证据路径、阻塞与下一步；恢复先查原run，不重复启动。


## 独立源数据覆盖假设

当前单阶段30轮使用原20份固定源腐蚀，后10轮重复01至10。新假设只改变每次访问的源噪声残差符号，提高固定条件下的数据覆盖；保持原SIG-only G266951、四项物理损失、actualSIG alpha1/输出梯度cap0.5、初始化、学习率和30轮/7410步。不是原20261017 F成对RDrop梯度方差假设，也不把它的FAIL改为通过；本次每批仍只有一次G前向，不作成对损失平均。

原R07 teacher代码已验证实际Action对无噪信号的重放。令s=A(clean)、n=received−s，独立Bernoulli0.5请求翻转成s−n，再以c=sqrt(原接收功率/新接收功率)恢复原接收RMS，同时使用cA作为全部源探针目标、c²variance作为方差目标，clean恢复目标保持。残差功率≤1e−12原接收功率的数值clean跳过。没有读取TX、V、target、F类别或场景标签来决定增强。

保持已实现的信道/锁定/AGC条件后改残差再重归一化，不宣称等于重新抽取完整原生模拟器或精确保持条件AGC分布。翻转前残差二阶能量与伪协方差保持，新增增益按c²改变并同步调整源目标。实际记录请求/有效翻转、gain、功率比及额外A(clean)访问，不把符号变体计为更多独立物理记录。专用NumPy2026103107不消耗原shuffle2026103004或Torch/CUDAprobe2026092701随机流。

原F50/source4df7/defaulthead冻结，source clean实际加权SIG40.32MB只读复用，新F缓存/VJP/数值复制0；历史63000前缀/0.847077秒与F50/50轮11100步的监督知识和成本保留，F-guided、不称F-free。旧SIG G不是本模型祖先。训练新增A(clean)1.89m访问，调整后的两个action probe合计3.78m；online F前缀/输入VJP各1.89m，完整F后续路径0。增强只在训练目标前执行，完整G仍266951参数、两prior/四CG/27FFT推理，实际时间和显存待测。

3项聚焦检查通过；已验证47基底命名空间下的actual factory266951、适配器/RNG隔离、单E30/all7 worker-score路由和完整入口通过，唯一P0/P1 PASS。首步真实2个非clean L_s（按缓存variance>1e−8选固定前2个，不读TX）强制[True,False]，检查实际增强与缩放Action/probe/variance/RMS、原F/cache、guidecap、coarse/action/prior全部梯度、严格E0和原生/优化推理及F原头状态；PASS立即继续。smoke G8/fullF4/prefix2/VJP2/源Aclean4/源Aprobe8，0TX/V/target/优化。

G从零固定最终30轮后，自动同卡profile/source1800×7/target168000×7，每role全七景预测固定后独立CPU truth-last；源表现差也不取消新G正式测试。主对照SIG47591、次对照pureG8cf及默认DURA/R17；六星地各≥137216正确/81.6761904762%、clean≥142256、R17每景最多损失1680和旧低城区70%分开报告。共享GPU单次墙钟不当作因果加速；默认DURA及Phase2/3不变。


## 实际启动核实

VERIFIED RUNNING。执行9925a6c6c4e73e3fa1f76f9b2c4105bfbfa91801，controller1774712/worker1774802/GPU4实际PID/CWD/完整命令、空CUDA控制器与CUDA4 worker、计算PID和日志增长已核实。启动前free22883MiB/已有1计算PID，新增1个符合总数≤3。首步真实2非cleanLs强制正负两路，实际RMS/Action-probe/variance/原F和SIGcache/guidecap/三G模块梯度/严格E0/native-optimized全部PASS。已完成2/30轮/494步；本轮requested=0.498095、effective=0.423016、功率比=1.000000000，数值clean跳过。SIGcache40.32MB只读复用，句柄核对与mmap打开0.054573秒（不是完整数值读完耗时），新F缓存/存储0；F50上游知识/原缓存成本披露。固定30后自动全部七景与独立truth-last，当前没有正式性能结论，默认保持。


## 完整训练与正式结果

VERIFIED / ANALYZED。原SIG-only主体266951参数，G从零单阶段30轮/7410步，L189000/U1701000，无G祖先/新可训练分支/推理增强。源残差符号请求944502、有效802880、请求clean跳过141622；全30轮独立重放专用NumPy2026103107和shuffle2026103004，计数一致，重复视图有效符号变化267569，不冒充新独立物理数据或完整模拟器重抽。源receivedRMS与cA/c²variance一致，额外Aclean1.89m/调整后Aprobe3.78m，原四物理项及SIG alpha1/cap.5保持；旧F成对梯度方差FAIL保留。clean SIG40.32MB只读复用/新缓存F0，历史63000前缀/0.847077秒与F50/11100监督知识披露，F-guided；online前缀/VJP各1.89m、完整F后续路径0、F状态不变。每role全七景预测固定后truth-last，126正式/105源V/252双对照配对分组/14IQ误差独立复算；原SIGcache证明复用，未重复NN检查。六星地较原SIG提高0/6、纯G提高2/6、默认DURA提高0/6。clean 84.866667%，low urban 67.717262%；低城区较原SIG -1.117857、纯G +0.344643、默认DURA -1.172619个百分点。六景3pp通过0/6，clean严格保持=True，R17逐景1pp=False，旧低城区70%=False。G训练及保存1080.461290秒，含worker准备1081.536184秒，控制器至自动全量分析1263.754893秒；25.068072m ConvLinearMAC/2prior4CG27FFT。共享GPU单次时间不是因果速度比或同质量效率证明；默认DURA保持，完整目标ACTIVE，有限模型不认证恢复极限。

详见[完整分析](analysis.md)、[训练与七景图](training_and_formal.png)、results.json及transfer_diagnostics.json。controller1774712/worker1774802自然结束，无目标反馈或选择性重跑。
