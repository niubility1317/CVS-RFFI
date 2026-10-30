# DURA两行冻结权重的零重训评估恢复

- run_id：`20261030-phase1-DURA-prior-anchor-s2026092701-m2-r02`
- group_id：`phase1-dura-prior-anchor-performance`；类别：`analysis`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：ANALYZED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

ReusebothoriginalE20/4940frozenweights;fixreaderconstructioninference_modeversioncounter;0newtraining;fullsource/target7viewsandindependenttruthlast

只读原r01的continuation/prior_anchor E20/4940、各63000×20曝光的final.pt，源checkpoint代码固定`44bb16ed08b29e7e4ee61c21e1e1287d18791c75`。源模型/权重/损失/预算不变，不重训，不移动或覆盖旧产物。原故障发生于postfreeze测速的F Sinc参数版本缓存，尚未进入正式prediction。

修复仅将评估函数外层构造环境从inference_mode改为no_grad，实际InferenceEngine仍FP32/TF32off、graph/reuse。四个精确回归先失败后通过，一次P1定点复审PASS。源权重commit与本恢复执行commit分别记录，不能改标源训练身份。

launcher第一步两条真实Ls与已冻结continuation/原F50 defaulthead无query smoke，随后按已登记资源自动补两行同卡成本、source1800×7和target168000×7，每组七预测完整后独立CPU评分。每行读取对应原sourcecomplete及严格customschema metadata，全部目标结果仅汇报，不回流原训练或单阶段候选。当前goal GPU worker合计最多2，其他健康owner保持。

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

## 本轮三版本完整结果与完整日志分析

VERIFIED。两行续训的原E20权重完整复用，r02新增训练0；两行source1800×7和target168000×7均闭合，控制器/worker自然退出。另一个从零E20版本也完成同矩阵。每行全七prediction固定后独立CPUtruth-last，之后同row再次按logits与truth复算全部126个总体/TX/RX/day分组；三行共378个分组一致，不把source成绩或单景均值代替正式结果。

以下均为同一原F50 seed2026092701/defaulthead、原168000物理样本的准确率百分比；不是当前Tri-SIG-TIME SSL或Phase2/3成绩。raw clean=84.676190%，故3pp门槛为每个星地场景≥81.676190%。

|场景|默认DURA v1.0|常规续训|先验监督|从零单阶段|
|---|---:|---:|---:|---:|
|clean|84.677976|84.627381|84.716667|85.402381|
|practical_high|82.825000|81.980357|81.872024|79.320238|
|practical_mid|81.560714|80.555357|80.513690|77.214286|
|practical_low_suburban|80.058333|78.991071|78.881548|74.964881|
|practical_high_urban|81.745833|80.899405|80.850000|76.856548|
|practical_mid_urban|75.486310|74.345833|74.403571|65.692262|
|practical_low_urban|68.889881|67.482738|67.857143|56.041071|

|版本|新增训练分钟|继承组件轮次/新增轮次|最终源IQNMSE|训练峰值allocated MiB|原R17逐景1pp容忍|六星地距rawclean≤3pp|
|---|---:|---|---:|---:|---|---|
|continuation|18.266|120/20|0.0209694|1024.02|True|False|
|prior_anchor|15.851|120/20|0.0211513|1022.02|True|False|
|scratch_joint|13.333|0/20|0.0401410|1034.42|False|False|

结论：三版都未完成六景3pp目标。常规续训七景均低于默认版；先验监督仅clean高0.038690pp，六星地均低于默认版。prior_anchor相对常规续训的lowurban高0.374405pp，但相对用户固定默认版仍低1.032738pp；不能只挑有利对照称成功。从零版clean85.402381%而lowurban56.041071%，证明可去掉祖先训练链运行，但没有保住星地质量，不能替换默认。

完整读取三行各20条epoch记录，共60轮/14820更新，没有抽样分析。常规续训/先验监督最终IQNMSE分别0.0209694/0.0211513；内部prior NMSE从对照0.2017508降至先验监督0.0313838，约下降84.44%，未转化为星地识别收益。对照全损失包含teacher权重日程，不能仅看loss总值下降当收敛或泛化改善。两行原stdout各一个已知postfreeze版本计数traceback，无NaN/OOM/Killed训练信号；恢复路径已核实成功。

从零版源IQNMSE从E1的0.7690035降到E20的0.0401410，最后五轮仍有拟合差距；初始峰值LR2e-4沿用了续训设置。依据这份source曲线，在读取该从零版正式分数之前已登记下一匹配从零E20、只改峰值LR到1e-3的独立run。未从目标成绩选父或回调参数，原从零权重不会用作下一run初始化。

新训练墙钟分别18.266/15.851/13.333分钟，GPU/并发负载不同，不能将跨卡差异直接解释成因果加速比。CPU固定batch256成本诊断中位数1.535041→1.186435秒、减少22.71%仅说明该CPU步骤；完整同精度训练效率仍未证实。参数统一266951、FP32参数1067804 bytes、ConvLinear MAC25068072、两prior/四CG/27FFT不变。匹配同卡G/G+F、B1/B128的median/P95/吞吐和共享显存见results.json；共享负载不与历史独占4.531972ms直接算加速比。

默认DURA v1.0保持，原R17逐景1pp是容忍上限，不等于优化目标。准确率、NMSE和净纠错均不单独证明每个TX细节真实恢复。完整逐TX/RX/day、波形及checkpoint产物保留在相应local_artifacts及原remote run，不提交IQ、模型或日志。本目标仍ACTIVE。
