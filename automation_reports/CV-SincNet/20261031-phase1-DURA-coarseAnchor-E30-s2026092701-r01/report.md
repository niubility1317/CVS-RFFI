# 弱粗恢复监督：保留原特征指导的单阶段30轮对照

- run_id：`20261031-phase1-DURA-coarseAnchor-E30-s2026092701-r01`
- group_id：`phase1-dura-weak-coarse-clean-anchor-SIG-E30`；类别：`cvs`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：ANALYZED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

原G266951/四物理项/实际SIG指导保持，仅加固定小权重coarse-clean NMSE；源梯度统计一次定权，无新模块、教师前向或目标训练，完整七景。

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


## 固定弱粗恢复监督对照

G从零单阶段30轮/7410，原266951参数、source01..20后01..10、model/CUDAprobe2026092701、NumPyshuffle2026103004、AdamW.001/warm247/cos2e−5/WD1e−4/clip1/FP32/TF32off保持。原四物理项和真实SIG指导alpha1/输出IQ梯度cap0.5不变，唯一差异为全部30轮附加alpha×coarse对源clean的逐包NMSE，alpha=0.02071440650605139。原coarse已在同一G前向计算，新增项没有额外G/F前向或训练参数VJP，也没有新推理模块、参数或上下文输入。

系数一次由历史源noVar E20、完整32批的coarse-clean/finalIQ参数梯度范数比中位12.068895139571893，按0.25/中位比确定。0.25为固定方法设置，不是最优性结论；旧checkpoint、原finalIQ分母与当前SIG模型/完整目标不同，不能把它说成当前训练的实际梯度比例或动态上限。旧梯度分离主假设18/32负cos、未达24/32的FAIL保持；本版不分离梯度、不增加预训练或继承旧G。源clean硬替换coarse曾在repeat E30小幅改善、在默认DURA明显损伤，说明coarse可能是联合潜在表示；因此弱监督是否有效只由新完整训练和正式测试判断。该源证据不保证真实TX或跨RX收益。

目标为原合法source clean IQ，无R17近似IQ教师或新F目标。源F-only真实加权SIG160缓存63000×160/40.32MB只读复用、新缓存F/数值字节0，历史63000前缀/0.847077秒及F50/source4df7/50轮11100步知识成本保留，F-guided。在线prefix/VJP各1.89m、完整F训练0，F参数/state/原头保持。G祖先0，0直接TX/V/target/classloss/另训头/sourceRX输入；目标仅received逐条推理，不做query关系统计。

3项聚焦测试及actual factory/单E30七景路由/有效objective和targetbuilder hook/原物理4-SIG与新aux权重/随机流隔离/help/历史系数精确匹配检查通过。唯一独立P0/P1 PASS。首步真实2个非clean L_s检查实际F/cache、原SIG cap、新aux仅coarse有限非零导数、全部G模块梯度、严格E0/nativeoptimized/F冻结。G8/fullF4/prefix2/VJP2，额外aux参数VJP2仅smoke，PASS立即进入训练。

固定final30后自动同卡profile、source1800×7和正式168000×7，各role全部预测完成后独立CPU truth-last。源差不能取消正式测试，不按目标成绩改系数、轮次、候选排序或选择性重跑。主对照原SIG47591，次pureG8cf/defaultDURA/R17；六景各≥137216正确、clean≥142256、R17逐景最多少1680，旧lowU70%单列。完整126正式/105源/252双对照配对分组/14IQ及30轮loss组成/曝光/视图/cap/aux固定权重复核。

GPU4/6/7/free≥12000MiB/existing≤2新增1总≤3，own1/card、goal≤2GPUworker、ownCPUtruthscorer≤1，其他owner不变。一次preflight/不可覆盖输出/单归档SHA/远端compile/实际PID-CWD-argv-CUDA-compute-loggrowth。默认DURA与Phase2/3保持，完整目标ACTIVE，既有曝光工程基准不称新盲测。


## 实际启动证据

VERIFIED / RUNNING，尚无性能结果。执行28555c9cdc8870930dffe295fd466111c97c0091；真实2非clean L_s smoke PASS后进入GPU4 worker1884426/controller1884336，已核实1轮/247步、两日志增长。实际CWD/完整argv/commit/CUDA/compute PID核对，准入前1个compute PID/剩余23271MiB，新增1个。归档SHA比较与远端编译通过；固定final30/profile/全部七景自动truth-last保持。详见launch_evidence.json。


## 完整训练与正式结果

VERIFIED / ANALYZED。小权重粗恢复监督G从零单阶段30轮/7410步，L189000/U1701000，无G祖先；完整G266951参数、推理图、原四物理项与SIG alpha1/IQcap.5保持。只增加全30轮alpha×coarse-clean逐包NMSE，alpha=.02071440650605139=.25/历史源32批梯度中位比12.068895139571893；该统计是旧noVarE20/coarseclean对finalIQ，不称当前SIG训练实际比例或cap。旧梯度分离主FAIL18/32未达24保持，本版不detach/预训练/复制G或用R17IQ教师；aux直接只更新coarse，原联合路径保持，无新G/F前向、训练参数VJP或推理模块。源SIG40.32MB只读复用、新缓存F0，历史63000prefix/0.847077秒与F50/11100源监督知识成本保留，F-guided。训练prefix/VJP各1.89m、完整F后续0，F不变，0直接TX/V/target/classloss/新头/sourceRX输入。每role全七景固定后truth-last，126正式/105源V/252双对照配对分组/14IQ独立复算；30轮loss/alpha/曝光/视图/cap核对。六星地较SIG提高0/6、纯G提高2/6、默认DURA提高0/6。clean 84.612500%，低城区 67.280357%；低城区较SIG -1.554762、纯G -0.092262、默认DURA -1.609524个百分点。六景3pp通过0/6，clean严格保持=False，R17逐景1pp=False，旧低城区70%=False。G训练及保存1020.983031秒、含worker准备1022.027162秒、控制器至自动分析1194.541898秒。ConvLinearMAC25068072/2prior4CG27FFT，同卡G+F B1默认4.591921ms、本版4.587787ms。共享GPU单次墙钟不证明因果加速，coarse更似clean不等于真实身份增强，有限模型成绩不认证物理恢复极限。默认DURA保持，完整目标ACTIVE。

详见[完整分析](analysis.md)、[训练与七景图](training_and_formal.png)、results.json和transfer_diagnostics.json。controller1884336/worker1884426自然结束，无目标反馈或选择性重跑。
