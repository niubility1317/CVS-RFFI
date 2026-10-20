# 冻结 R17 两版精简模型的目标域对照 R05

状态：RUNNING。用户最新确认性能以目标域测试为准，逐场景准确率相对原R17下降不超过1个百分点（80%至79%，不是相对下降1%）。固定评测R04两版source-final E20/4940权重，每版168000物理样本×clean和六practical，原F50 seed2701/4df7/联合训练默认头不变。按整数正确数逐项判断，每视图最多减少1680条正确样本；不以均值、舍入或波形指标覆盖超限。low urban≥70%及其他环境接近clean的长期目标另报，不自设“接近”的容差。已暴露基准，不称新盲测。

|预先固定的目标行|完整参数|相对原4665073|新增训练|
|---|---:|---:|---:|
|step_norm_task|3118001|减少33.1629%|0|
|activation_task|3775105|减少19.0773%|0|

R04两版均完成完整源训练和独立源评分；源准确率各视图均在1个百分点内，但原波形护栏失败，原run如实保留source_qualified=false/selected=null/target_rows=[]。这不证明目标识别失败。按最新用户要求建立独立固定双行目标比较，不伪造原源资格、不追溯改写旧结果、不修改权重或重新训练。两行、最终轮次、参数和目标范围在任何新压缩模型目标结果前固定；逐行报告预先规定的验收结果，不回流训练、超参、候选重排、追加候选或选择性重跑。

权重只读加载R04原final；严格校验R04方法commit a310b3b5ff30e23584e7bc950e8a0ff96ac685fb、E20/4940/L126000/U1134000、0新TX/V/target训练和原F/默认头0更新，沿用原R17/source-only coarse/action与原F50数据契约。R04使用冻结源监督F知识，不能称F-free无监督。prior原20+新20=40，冻结coarse20/action20共80继承串行pass，F50原50/11100及R04额外Fteacher/student各1260000前向与输入反传成本沿用原记录；本轮0新训练、0校准/头拟合/目标状态更新，2Ls smoke单列。

新comparison_plan明确两行目标比较依据；reference仅在计划匹配后打开既有formal IDs/IQ。不创建伪造的source_decision合格记录。恢复器推理仅当前IQ，3步×2CG与原coarse/action/variance保持；原F只作下游识别，0TX/RX/场景/clean/教师/跨帧输入或状态更新。各row七视图IQ/feature/logit全部固定后，独立CPU scorer才连接truth及clean，报告accuracy/MacroF1、TX/RX/day/confusion、raw repair/damage/net，以及full256/prefix232/tail24波形与配对clean特征差距。独立聚合再对照原R17精确正确数，保留零下降诊断。原source延迟/显存profile作为已有证据，不重复训练或重跑旧目标预测。

28项聚焦检查PASS，唯一独立P0/P1审查限新适配/调度层。launcher第一步实际R04两份final+原F50+2Ls CPU no-query smoke，严格加载、完整参数、恢复IQ/F输出有限；0TX/V/target/optimizer/state/head更新，PASS立即继续。不是目标性能证据。

N607普通账户，GPU4/5各一own target worker，free≥12GB/allowner<2，保留其他owner；R04训练/评分/保存预测分析实际进程均已退出。CPU2线程/max1ownscorer，ownscore.lock串行。一个不可覆盖新root、一次普通账户预检、一次发布归档SHA比较/远端新modulecompile、实际PID/CWD/cmd/GPU/日志增长核实。技术失败保留失败行，健康行继续，无自动重试；性能下降不作为技术重启理由。具体命令、环境/CWD、路径、GPU和预期产物见experiment.json。

发布前检查：本地28项聚焦测试PASS；新适配/调度层唯一独立P0/P1审查PASS，独立复跑24项合成测试，未访问真实数据、权重或远端。普通账户N607预检VERIFIED，原source/R17基线句柄与R04完整预算匹配，R04实际旧GPU进程0，GPU4/5各24124MB空闲，新输出/release不存在。实际R04-final+F50+2Ls smoke由launcher第一步执行。

实际发布 VERIFIED：方法commit `81c39cb908373d4e19b70ec8406f4083529da99a` 已推送且远端OID独立比较一致；一次归档SHA比较、新module远端compile/help通过。真实R04两版final/原F50/2Ls CPU no-query smoke PASS，完整参数/输出有限、0TX/V/target/optimizer/state/head更新。两个target真实PID/CWD/完整命令/commit/GPU4或5/日志增长及首个clean场景预测完成已核实。原R04源资格和权重均保持。全部七场景预测完成前不连接truth；当前尚无完整目标结果。

独立聚合分析已实现，5项合成检查PASS，覆盖固定两行都必须完整、真实句柄退出、缺失视图拒绝、运行时判定与整数正确数重算一致，以及多错1条跨过1个百分点边界的失败分支。分析不修改正在运行的模型、候选或方法commit。

两份冻结恢复器已取回本地并独立验证：共享版12777244字节/3118001参数，半秩版15475140字节/3775105参数；文件大小与远端一致，严格来源/最终轮次/参数加载通过，独立恢复图在合成零IQ上输出有限，0真实IQ/目标访问。文件与使用说明位于`local_artifacts/pamf_restoration/r17_target_compare_20261020_r05/models/`，不提交权重。checkpoint含完整coarse/action/prior，不含优化器或F分类器；推理输入为原生尺度float32 `[B,2,256]`。文件可用不代表目标性能通过，完整七场景仍在预测中。

非阻塞同row目标逐样本对照已实现，5项合成权限/完整性检查PASS。仅在两行固定目标预测与独立评分完成、实际进程退出后，验证原R17及两版全部3×7保存预测和物理IDs，再连接同一truth，报告原正确丢失/原错误修复/净变化、决策翻转、教师置信度与TX/RX/day分组。0新推理/原始IQ/权重读取/拟合/训练更新，不改变候选、1pp验收或方法commit；不是额外性能门槛，当前未执行实际配对分析。
