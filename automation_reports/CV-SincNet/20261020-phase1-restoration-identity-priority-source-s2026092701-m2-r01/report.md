# 三套冻结原 F 的决策约束强度消融

状态：RUNNING / VERIFIED。目标保持 low-urban≥70%，其他五种 practical 接近原clean（Agent暂按差距≤2个百分点解释），并分别报告恢复IQ与同条clean差距、接入冻结原基座的目标识别。任何源分数或NMSE单项不能证明目标达成。

## 源依据和固定两臂

已有源实验中，KL权重1的correction-only恢复使low-urban比旧feature参考提升2.50个百分点、mid-urban提升1.388889个百分点、clean不降，但平均波形NMSE增加8.84%，未通过当时5%界限，且未访问目标。保留该负结果，不追认、复用或重新晋级该权重。本次另立原R03独立初始化、联合训练两个恢复模块的匹配消融，使用三原F监督知识检验身份/波形取舍。

posterior03、posterior10仅后验权重为0.3/1.0不同，均平均三个teacher的逐样本标量loss；每teacher保留独立160维特征坐标，以同条clean特征cosine(权重1)与原默认头T=2/T² KL约束恢复IQ，另加IQ NMSE+0.1 log power。三套原F50 CROSS_RX种子2026092701/02/03与原联合训练默认头全部冻结，不另训头，不读取L/U TX真值，不平均坐标或投票。源soft targets含既有监督知识，因此是识别器引导的自监督恢复。

从原R03 coarse20/correction20独立分叉，新增联合E10/2470，组件各至30轮、60组件epoch、继承串行40+联合10=50 pass。seed2026092701/data2026101951，原R02前十轮63000条L/U、路由.15clean/.60residual/.25full、同shuffle；paired observed→clean与clean→clean等权，未生成clean槽不读取damaged。batch256，reset AdamW3e-5/WD1e-4，warm247、cos至3e-6、clip1、FP32，每臂L63000/U567000/IQ1260000。只用已完成single-F joint_posterior E10作为固定源指标参考，绝不作为初始化；新两臂训练成本严格匹配，teacher计算成本实测。

## 预先固定的资格和自动目标

两个新臂末轮固定，完整1800条源V/原F1/七视图预测后独立CPU评分。相对完成的single-F E10，low/mid-urban各至少+0.5个百分点，其他四场景最多下降0.5个百分点，原clean下降≤0.1个百分点。本次波形界限在任何新候选V评分前明确为1.15×原feature源平均NMSE0.028326367493718863，不能在新联合参考上复利。将15%设为新消融的前瞻界限，是为了覆盖既有源KL1的8.84%取舍；不修改历史5%失败，不代表用户已接受目标波形或最终达标。选择通过者的源urban均值最高、NMSE较低、登记顺序；都不通过则无新目标。

仅选定合格一臂自动对原不可变168000条clean+六practical、三套原F/default head分别argmax。全部七视图/恢复IQ/物理ID固定后，独立无CUDA CPU scorer才接入目标TX/RX/day和评分clean，报告准确率/Macro-F1、修复/损坏/净修复、逐F/TX/RX/day、距clean百分点差、full256/prefix232/tail24 NMSE和logFFTpower均值/中位数/P90。practical推断只见当前receivedIQ，不见clean/目标真值/其他视图/状态更新。一个恢复器种子、三个现有读出，目标基准已暴露，不声称三恢复器种子、新盲测或在轨结果。

## 资源、验证和停止

26项聚焦检查通过，包括实际0.3/1.0改变loss和IQ梯度、预算/权限、三F坐标/冻结、源选择、全部七目标视图/truth-last边界。一次独立P0/P1审查PASS，无发现。N607普通账号，GPU4/5各一条本run训练worker，总owner<2、free≥18GB才获取训练容量，不改其他健康任务。此时已有多teacher run继续GPU0/1原计划；新run训练/预测后等它的complete/failure，才进入新run单CPU评分锁，避免本对话独立CPU scorer并行。目标在本run两源worker退出后才启动选中卡一条worker。

一个code/config commit、一个不可覆盖root、一次预检、一次发布归档SHA比较、一次远端compile。launcher第一步原R03+三原F两条Ls no-query smoke，检查两恢复模块有限非零梯度、F全部冻结、原头到IQ梯度、新checkpoint往返、0目标/0更新，PASS立即继续。随后核实实际PID/CWD/完整命令/GPU/日志增长。数据/来源/commit/预算/输出冲突、非有限、OOM或预测失败时保留产物、停止对应技术失败且不自动重试；不因指标或另一run进展热改健康任务。具体命令/环境/路径/矩阵/产物见experiment.json。

一次只读预检核实GPU4可用24124MB、GPU5可用22905MB；GPU5已有一个其他owner进程，满足全owner少于2且可用≥18GB的获取条件。保留其进程和产物。


## 启动核验

方法commit：`de3118c724026869581a9cf1975b0d1314c96d6e`。26项聚焦检查通过；一次独立P0/P1审查PASS，无发现。一次发布归档比较及远端compile/import通过。真实R03＋三套原F的两条L_s smoke证实coarse和correction均有非零有限梯度、三套F均无梯度、新权重往返一致，0目标访问、0更新。PASS后两臂直接继续训练。

独立后态已核实两条实际worker的PID/CWD/完整命令、物理GPU4/5归属和从0到首轮的日志增长。当前进度：

- posterior03：2/10轮、494/2470步。
- posterior10：1/10轮、247/2470步。

两个源参考特征缓存均为63000条L/U×三套F独立160维坐标，没有TX真值读取。当前没有源V评分或新目标性能结论；只有固定源规则选中的通过臂会自动进入目标七视图／三原F验证。
