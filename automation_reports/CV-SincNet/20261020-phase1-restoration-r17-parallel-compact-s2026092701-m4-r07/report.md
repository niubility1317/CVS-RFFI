# R17 四种精简结构并行验证 R07

状态：RUNNING。用户最新要求服务器空闲时一次性发布多个、快速迭代。本轮在健康 R06 两行仍于 GPU4/5 训练、尚无新目标分数时预先固定四种结构，加用 GPU2/3/6/7，每卡一个 own worker，总共六路 own GPU；保留其他 owner。四行完整训练后自动 source 七视图诊断及168000物理样本×clean和六practical目标测试，逐行预测全部固定后独立 CPU truth-last 评分。R06/R07 共用 R06 score.lock，CPU2线程、跨六行最多一个own scorer。

|固定行|完整恢复器参数|比原4665073减少|秩比例|原1x1权重跨步共享|GPU|
|---|---:|---:|---:|---|---:|
|shared_three_eighths|2702125|42.0775%|0.375|False|2|
|shared_eighth|2463685|47.1887%|0.125|False|3|
|shared_quarter_pointwise|2471609|47.0189%|0.25|True|6|
|shared_eighth_pointwise|2352389|49.5744%|0.125|True|7|

只从原 source-selected R17/cd977448/E20/4940 初始化与蒸馏，不加载 R04/R05 或仍训练的 R06 学生。原三步长卷积在 CPU 取均值后做平衡截断 SVD，共享左右因子权重；另两臂在分解前共享原1x1卷积权重均值，所有逐步bias/GN和其他状态保留，避免误把分解产生的1x1当原短卷积。0初始化IQ/F/V/target/优化更新。原coarse/action/nativevariance、三步各两次CG不变；完整成本含冻结上下文2157547参数。新秩和共享方式来自原R17源权重代数，不使用目标结果设计/选模/重跑。

数据与预算同R04/R06：R07原source缓存63000=L6300/U56700、epochs1..20、15%clean/85%residual，batch256/model2701/data2026102001；AdamW2e-4/warm247/cos2e-5/WD1e-4/clip1/FP32/noTF32，E20/4940。固定损失10×三步IQNMSE_to原R17 + 原F50/defaulthead T1 KL + .5×unitF160距离平方，task系数1。F教师参照为同receivedIQ经原R17恢复，不是clean；sourceclean仅detached能量分母floor1e-4/日志。只优化精简prior，0新TX/V/target训练、F/defaulthead/context/teacher更新与头拟合。使用源监督F知识，不能称F-free无监督。每行L126000/U1134000，R17/Fteacher/Fstudent各1260000前向，student路径输入反传；prior继承20+新20=40，coarse20/action20合计80继承串行pass；F50预训练50/11100及教师成本分列。

source/V/波形仅诊断，固定最终E20，无源资格门槛/早停/候选重排；每行完整源诊断后即自动目标测试，不等其他行训练完成。相对原R17每场景最多下降1个百分点，168000样本允许最多损失1680条正确，按整数判断，不以平均/舍入覆盖超限；零下降另诊断。报告accuracy/MacroF1/TX/RX/day/confusion、raw repair/damage/net、全帧256/主体232/尾24 NMSE、FFT logpower/配对clean特征差距、完整参数/序列化字节/source推理成本。low urban≥70%及其余环境接近clean的长期目标分别核对，不自设“接近”的阈值。已暴露基准，不称新盲测/全局最简/Phase2或在轨证明。

40项合成/协议检查PASS，覆盖四模型参数和alias、1x1均值与独立bias/norm、原teacher保持和梯度、严格复载、最终预算/权限、目标前完整源诊断、七视图truth-last及错误行隔离。唯一独立P0/P1审查后固定code/config commit。launcher第一步实际原R17/原F50/2Ls CPU noquery smoke，检查输入梯度、冻结F/head/context/teacher、各prior梯度、严格重载逐位输出；0TX/V/target/optimizer更新，PASS立即四行训练，不再等待确认。

一次普通N607预检核实新root不存在、原数据与继承句柄、四卡空闲和旧R05实际句柄退出；健康R06保持。source启动free≥18GB/allowner<2，沿用共享capacity lock；新不可覆盖run/output/release，单归档SHA比较/远端compile，PID/CWD/完整命令/GPU/loggrowth实际核实。契约/nonfinite/OOM/技术错误停止受影响行并保留，健康行继续；无自动重试或回退，性能负结果不作重启理由。source profile先CPU加载、删除payload/optimizer再上GPU，报告resident/peak/增量peak；受同卡owner和顺序影响，不直接称部署加速。具体命令、环境/CWD、路径、GPU和预期产物见experiment.json。

发布前证据：40项本地合成/协议检查PASS，唯一独立P0/P1审查PASS，无确定P0/P1；独立复跑结果已记录。普通账户N607预检VERIFIED，GPU2/3/6/7各24124MB空闲、新root/release不存在；R06健康两行仍在GPU4/5，观测已完成E9，不停止或修改。旧R05实际句柄已退出。真实checkpoint smoke由launcher首步执行。

实际发布 VERIFIED：方法 commit `f03015d75428f156430cff60633e1c99ec4983b4` 已推送且远端 OID 独立比较一致；一次归档 SHA 比较/远端 compile/help 通过。真实原 R17/原 F50/2Ls CPU no-query smoke PASS，恢复与 F 输出有限、F 输入梯度和各步 prior 梯度有限非零、严格复载输出逐位相同，0TX/V/target/optimizer/F/head 更新。四个实际训练 PID/CWD/完整命令/commit/GPU2/3/6/7/日志增长已核实，各完成 E1/247；这些是训练执行证据，不证明目标性能。固定配置继续到E20，再逐行自动源诊断和全部目标测试。原R06健康GPU4/5继续，跨六行共用R06score.lock。

独立汇总工具9项合成检查PASS，覆盖四个固定row、20轮全日志及真实曝光、损失算术、各七视图、分组混淆矩阵、独立assessment与整数1pp判断的一致性、profile排除optimizer和显存基线差。工具不修改运行中模型、配置或方法commit；当前不读取新目标准确率。
