# 原头5维与新增5维：源IQ梯度冲突及等步长诊断

- run_id：`20261031-diagnostic-DURA-source-headAux-s2026092701-r01`
- group_id：`phase1-dura-source-head-aux-gradient`；类别：`diagnostic`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：ANALYZED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

复用固定pureG源512×01/20 IQ，分解精确head5/aux5梯度并比较物理IQ+cap1的head/joint等IQ预算一步；0优化/新G/TX/V/target。

## 数据、seed与模型来源

实际数据契约、权限例外、完整seed角色、checkpoint来源和选择规则见experiment.json。
逐行配置通过config_ref/resolved_config_ref定位；待补项必须在对应生命周期补齐。

## 执行与存储

命令、环境、CWD、commit、launch owner、输出和日志路径见experiment.json。
实际PID/GPU、读取时间、remote readback、失败或替代关系在此追加，并用record命令记录证据指针。

## 结果与覆盖

完整SOURCE_ONLY结果见analysis.md/results.json；本次无新G训练或正式预测。
源域训练完成、预测完成、评分完成及协议有效性分别陈述。不得用总索引或旧状态证明当前运行。

## 交接

记录已完成、当前run/commit、证据路径、阻塞与下一步；恢复先查原run，不重复启动。


## 固定源域矩阵与判定

固定原pureG8cf/source512（128L＋384U）×缓存01/20，原F50/defaulthead、source10参考/beta0.25/alpha保持。分解head5与aux5实际IQ梯度，再比较物理IQ＋cap1的head/joint一步，二者每条修正L2预算同为min(0.01×clean L2，0.1×恢复误差L2)。只使用配对源IQ，属于离线oracle，不是部署路径或G训练。

只有两视图均满足：全部方向定义；head/aux负cos比例≥25%；joint/head方向cos中位<0.9；head步骤与clean F判断一致数较joint至少多4/512；5个源RX均不损失；head平均NMSE≤baseline×1.02，才支持另登记原头5维独立训练对照。失败不修改阈值/步长/alpha/beta，不搜索其他父模型。没有TX真值或正式query读取，不把clean F一致率当真实TX准确率。

3项聚焦测试和实际模块入口通过；独立P0/P1通过。N607实测bool DLPack不支持，写出前统一转FP32已修复并通过该问题定点复审；defined保存0/1。首步真实2Ls原F/defaulthead/cache与全梯度分解smoke后直接继续。主F3072/输入VJP2048，smoke F4/VJP6，新G前向/权重加载/优化/TX/V/target均0。完整1024条数组与RX/day/L-U分组独立复算；本次无新G或正式测试矩阵。默认DURA及六景3pp/clean/同质量效率目标保持。


## 完整结果

VERIFIED / ANALYZED，SOURCE_ONLY，固定科学判定FAIL。source512（128L/384U）×01/20、CPU2、主循环5.400632秒自然结束。head/aux负cos比例19.53125%/20.3125%，两者均未达事前25%；joint/head cos中位0.820375/0.815878。相同IQ修正预算下，head相对joint与clean F判断一致数+3/−1，未达双视图至少+4，第二缓存还有源RX损伤。不据此启动删aux5的E30、不改阈值或步长。主F3072/输入VJP2048、smokeF4/VJP6，0新G/优化/TX/V/target，F状态不变；1024条几何、2048步及22分组独立复算。默认DURA与完整3pp/clean/同质量效率目标保持。

完整规则、数值与限制见[分析](analysis.md)。执行提交31d614135af641db4dfab54fcfebf91dc40b7741；PID1634503已自然结束，实际启动与全部产物独立读回通过。
