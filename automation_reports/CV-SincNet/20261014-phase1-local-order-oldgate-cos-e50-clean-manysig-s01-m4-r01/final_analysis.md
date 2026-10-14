# SIG_TIME 浅中层有界身份排序：首种子完整结果

固定modelseed2026092701、原ManySigclean168000/row、普通checkpoint默认cos头联合训练。首seed4row已完整freeze/predict/独立truthlast评分及全量日志/梯度复算，状态VERIFIED；本节不是三seed结论。

| 方法 | clean准确率 | Macro-F1 | 最差RX | 最差day |
|---|---:|---:|---:|---:|
| 旧门控＋cos历史 |80.6685%|80.1734%|70.6167%|75.4524%|
| SIG_TIME零辅助 |83.5452%|83.1080%|70.4708%|79.0190%|
| LOCAL_SUPCON |83.4583%|83.0104%|70.4708%|78.8786%|
| LOCAL_ORDER |83.6238%|83.2006%|70.8833%|79.2238%|

ORDER−SIG +.0785714pp：修正538/引入406/净+132；ORDER−SUPCON +.1654762pp：修正778/引入500/净+278；ORDER−旧基线 +2.9553571pp。SUPCON−SIG −.0869048pp，修正625/引入771/净−146。所有72个overall/RX/day/TX组指标、五项配对错误表及混淆矩阵在results.json。n=1不计算样本标准差或宣称稳定提升，最高单seed或其他读出都不替代原三seed84%目标。

核心实现26bfdf044caa1be3c966370248c3bc3372009943；首seed执行f34078d1a0775ef6fa66365689481ad5a1932e4c；全量独立复算9cd9a031652058936e381cd9ad3e85df539fd6c1。主结构为单路SIG_TIME，fθ195041＋普通类别W960，SIG池18976参数；三臂同架构、无新增拟合投影/分类头参数。ORDER/SUPCON直接作用既有t1/t2四格有序时序视图，原分类头仍由原L/U流程联合更新。NoExplicitCFO/internalcompensation、source角色6300/56700/27000、split392005和33+17旧batch_neighbor/ownEMA.999/cos保持。

三新模型各scratch50/11100，共150新增model-epochs33300updates；另计历史baseline50/11100。三臂每模型2750额外L90eval-with-gradient前向/247500包，合742500额外源包，原主L/U预算不变，无额外optimizer/headfit。每模型963900U包次，实际接受SIG15430、SupCon15422、Order15431，均约1.60%，仍为原随机batch旧门控而非PAIR覆盖。

完整复算：200row-epoch记录、33300实际新增native步、8250实际aux步、672000固定默认决定、72分组指标、1980实际层/cohort梯度记录（每记录含CE/ORDER/SUPCON/effective四向量）全部VERIFIED。raw PT/CP/IQ/预测/原始日志留远端既定run，未入Git；公开全部源epoch曲线、实际aux曲线和四声明epoch的梯度汇总，见results.json/results.png。

实际epoch1辅助关闭，所有effective梯度为0；epoch17/33/50 ORDER和SupCon在全部55cohort的t1/t2均非零，t3辅助恒0，classW依赖逐aux检查为None。epoch50 ORDER辅助/主L CE梯度比：t1 .9923%、t2 .5515%；SupCon为6.0912%/3.8006%。这里是实际主L与另一个合法L90cohort的梯度，不能解释成同物理包方向、全程累计贡献或跨层重要性。保持原固定系数，不因幅度较小追加扫描。

末epoch采样关系：SIG的t1/t2未满足率88.277%/81.474%，ORDER87.720%/80.304%，SupCon83.356%/71.454%。各视图4950源包次全部仍触及至少一个未满足关系；这是重复包次，不是4950独立物理样本。SupCon关系指标更低、梯度更强，却没有首seed目标收益，再次说明这些量不能替代泛化排序；ORDER仍有大量活跃项，也不宣称已经使源关系普遍满足。

三模型五个SIG池参数张量全元素数值更新；actualjoint160导出最大差0，W清零或删除不改变特征，诊断无参数/buffer更新、无新headfit。实际同seed初始body/W/参数相同、初始source函数同，最终ORDER−SIG L720zL2=1.592485。运行保留原GPU非完全确定执行，启用前的原始梯度已有细微跨模型差异；特征变化或单seed微小目标增量不能全部因果归于新loss，更不能推断新TX能力。

源困难关系补充：先前全L诊断的joint160中，源标签3→1及1→3分别有2704/4200和2406/4200未满足项，占全部13543项的37.73%；涉及2073/2100和1940/2100个两类源物理样本，支持“困难集中于若干身份对但端点覆盖广”，不能用少数异常包概括。三day及五RX均为训练源，非未见域证据；本矩阵保持均匀关系规则，没有按目标混淆定制或硬挖单个极端包。

扩种只按source：目标数字首次回读前，三末轮LCE .247704585/.248384086/.247982446，actual导出/更新/W独立/梯度预算/初态同以及ORDER−SIG zL2阈值均PASS。扩种决定在6d65f77a7659cb182bdb39852dc6c6b76001e81d冻结提交并上传，随后原样发布20261014-phase1-local-order-oldgate-cos-e50-clean-manysig-s0203-m8-r01；其后才回收首seed目标数字。02/03六新scratch50已真实启动，首seed未重训、参数/关系规则/门控不改。完整三seed仍pending，84目标未达；不改变健康任务或根据这次+.0786pp选择性重跑。

下一步只完成已登记剩余种子的自动预测/评分、完整三seed配对与错误增减分析。未得到完整结论前不叠加PAIR、层级、权重或PA重组，也不改分类头。
