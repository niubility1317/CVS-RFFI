# 粗恢复器与修正器联合优化

状态：PLANNED。目标仍为 low-urban 至少70%，其他五种 practical 接近同一原 F clean（Agent暂按差距≤2个百分点解释）。分别报告恢复IQ与clean差距、接入冻结基座后的目标识别，不能由源域或NMSE单项宣布目标达成。

## 固定消融与成本

从同一原 R03 独立分叉两臂 joint_feature、joint_posterior，都联合优化原 coarse 与 correction；原 F50和默认头完全冻结。两臂均使用原波形NMSE＋0.1log power、特征cosine权重1；仅joint_posterior增加原默认头T=2、T² KL，权重0.1。该权重来自上一条源域证据：KL权重1使low/mid-urban改善2.50/1.388889个百分点、clean不降，但波形NMSE恶化8.84%而未过固定源规则，未访问那条候选的目标数据。此次检验解除coarse梯度阻断能否改善这项取舍，不从失败posterior或已测目标feature权重继续训练。

新增联合阶段10轮／2470步、batch256、seed2026092701、data seed2026101951、reset AdamW3e-5/WD1e-4、247步warmup、cos至3e-6、clip1、FP32。两个组件各由20轮至30轮，合计60组件epoch，继承串行40轮＋联合10轮＝50轮；原 F保持50轮。上一轮是coarse20＋correction40＝60组件epoch，两者组件更新成本相同，但本轮新增10个联合优化pass，不能冒充相同optimizer步数／实测算力成本。两条新臂之间参数、预算、数据顺序与配方完全匹配。

每臂L/U优化曝光63000/567000，IQ例子1260000；observed→clean与clean→clean等权，原R02 clean/residual/full路由和前10轮缓存不变。未生成的clean槽不作damaged读取。F参考特征由合法L/U生成；恢复器不读取TX真值。使用已有监督F知识，因此称识别器引导的自监督恢复，不称整个系统完全无标签。

## 源规则与唯一目标候选

两个源臂均固定末轮、无V早停。每臂1800条V的clean＋六种practical全部预测后独立CPU评分。各臂相对固定旧feature参考须low-urban和mid-urban各≥0.5个百分点、clean相对原IQ降幅≤0.1个百分点、六practical平均NMSE≤1.05倍旧feature。原旧feature只作既定参考，其新增20轮不与本轮10轮冒充相同pass预算。

在通过者中按low/mid-urban平均准确率最高选择，完全并列取NMSE较低，再并列按登记臂顺序。选择和权重在任何新目标访问前冻结。若均失败则保留负结果、不访问新目标。只为选定一臂自动测试完整168000条clean＋六种practical、三套原F50默认头；不测试未选臂，也不根据目标重排候选。

目标全部七视图恢复IQ与三套F逐包argmax固定后，独立无CUDA CPU scorer才读取目标TX/RX/day与评分clean。报告accuracy/Macro-F1、修复／损坏／净修复、逐F／TX／RX／day、距clean百分点差，以及未作oracle对齐的full256／prefix232／tail24 NMSE与logFFTpower差距。clean只作为单独测试视图和评分参考，不进入practical推断。一个恢复器种子、三个现有F读出，不作三恢复器seed或新盲测／在轨声明。

## 发布和停止

21项聚焦检查覆盖joint两模块梯度、eval与父图一致、权限／预算、源选择和禁止未选臂target、全部七视图／三F／ID／类／IQ，以及提前truth/clean/状态更新。一次独立P0/P1架构及衔接审查无发现；E10数值预算在发布前固定并重跑聚焦检查，不增加重复审查。

一个code/config commit、一个不可覆盖root。N607普通账号，GPU0/1各一条本run worker，总owner<2且空闲≥12GB，CPU2线程、最多一条scorer；源两臂完成退出后，仅启动选定臂对应GPU的target worker。启动第一步真实R03＋原F两条L_s smoke，验证coarse/correction均有有限非零梯度、F无梯度、新checkpoint往返一致，PASS立即继续。一次发布归档比较、远端compile/import、实际PID/CWD/命令/GPU/日志增长保存。数据/模型/commit不符、输出冲突、非有限、预算或预测失败则保留产物停止、不自动重试，不停止其他健康任务。

具体命令、环境、路径、矩阵和预期产物见[experiment.json](experiment.json)。
