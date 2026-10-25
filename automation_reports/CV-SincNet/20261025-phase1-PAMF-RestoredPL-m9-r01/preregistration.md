# 保留普通 PL 主项的恢复域半监督九学生实验

PLANNED。用户授权持续研发、三种子和全部显卡；已完成先分析计划及两项冻结源可行性诊断。本 run 独立执行，不接管、停止或修改既有健康 owner。默认 PAMF/DURA 历史权重保留，新方法仍是独立候选。

## 源冻结决定与匹配矩阵

固定 F0，模型 seeds2026092701/02/03；每学生严格加载自身原 ordinary-PL E30（scratch20+own10），新增20轮/4440步至E50/11100，累计13L+37LU。原L6300/U56700/V27000、sourceRX[1,3,4,6,8]/day[1,2,3]、TX及物理角色不变；默认分类头随骨干联合训练。使用原生分段入口与相同优化器构建、cosine及CENTER_UNITS，不宣称从单个连续优化器状态无缝恢复。

| 方法 | 原普通PL主项之外的学习 |
| --- | --- |
| PLControl | 无新增分支；新跑匹配E30+20控制 |
| RawSoft | 额外raw U软标签监督 |
| DURASoft | 将RawSoft新增输入换成同物理G(H(U)) |

原L六前向ordinary/raw RDrop、As两次真实标签CE/As KL0、浅中层跨RX及全部原损失保留；原U主项仍为自身eval的raw argmax/confidence≥.95、权重1、完整batch归一的普通PL。新增三教师规则只作用于B/C额外分支，不替换原raw PL主项。

B/C共用原三份冻结E30对raw U的平均概率；三者同类且各confidence≥.95才接受。新增损失为两次student前向的软CE先平均，再按整个U batch平均；默认头/全骨干反向。新增权重固定 **0.12618073797689983**，首444个新增step线性升温。该剂量由已登记12个源L/U批的预定物理梯度范数公式确定，未读取新目标分数。三模型源V共识精度99.9683%只是U质量代理，U真值始终隐藏。

40dB恢复后噪声在预定9个源单元中6个预测稳定性未通过，因此D按原规则排除。B/C均不做后增强、不使用G输出置信度门控、不强拉波形或全特征、不另训头/EMA/G。

## 缓存、预算与来源

先用L2真实权重无query smoke，再一次性缓存全部U：3个固定源practical residual家族low_suburban/mid_urban/low_urban及对应同契约G-LU恢复，FP32/TF32off、reuse=True/graph=True/native=False，无二次归一化。源信道seed20261016421、硬件seed20261016423、namespace PAMF/restored_dose/U/fixed，与源剂量诊断一致。物理索引与U角色严格配对；cache worker仅接收最小source context，不含目标句柄；不读取V/U标签。

共享cache教师前向170100次、G前向170100次，恢复IQ约348MB；成本单列。每个学生增量U曝光1134000，B/C新增student前向2268000次。每组3学生共享3份E30：父90唯一epochs，9条新尾段180epochs，合计270唯一F epochs；单学生自身累计50外还有另两共享E30教师成本，不能声称9个完全独立50epoch训练。

加载每个父/教师前检查原E20→own10实际headers、payload native U/EMA关闭及同契约来源；G沿用已核实同契约F-free/TX无标签来源。复用VALIDATED_ONCE角色，不新增数据哈希或逐文件封印。

## 自动测试与真实验收

共享cache完成后，实际E30+cache执行完整L768/U256/恢复分支512前向的有限梯度默认头backward smoke，无optimizer/query；PASS直接进入9条训练。每条冻结后立即测试原三人群cross_date30000、cross_rx126000、cross_both42000，raw/同一G-LU×clean+六practical。沿用原目标received/G-LU IQ缓存，不重新抽目标信道。单行全部42个组合预测固定后独立CPU truth-last评分；score配置不包含source模型/IQ/G句柄。

主目标为目标RX所有日期168000记录/seed，跨日30000诊断另列。三seed均值同时满足：raw clean≥原supervised/ordinaryPL两对照较高者+1pp；raw六景>较高者+1.5pp；同G-LU六景>两个对照。用confusion整数和Fraction验收，G clean和逐场景/种子/分层/配对SD完整保留。新PLControl不能替换原目标对照；SD不等于显著性，已暴露基准研发背景保留。任何单行分数都不回流到其他配置、权重、排序或选择性重跑。

## 最小发布与停止规则

代码/config身份为本次Git commit，runtime记录实际值；精确矩阵唯一维护于experiment.json。4条针对角色、缓存物理索引、源决定、原主项/预算/头和升温的测试及编译PASS；一次独立P0/P1审查PASS。源码与科学配置在独立不可覆盖release/run冻结；一次archive传输SHA比较、远端编译及启动后PID/CWD/argv/GPU/loggrowth核对。

命令：`PYTHON -u -m experiments.pamf_restored_pl.runner dispatch --config PROJECT/releases/20261025-phase1-PAMF-RestoredPL-release-r01/runtime.json`；CWD为该release。run root为PROJECT/runs/20261025-phase1-PAMF-RestoredPL-m9-r01，私有本机runtime位于local_artifacts/pamf_ssl_research_20261025/restored_training。

N607普通账户，seed01 pool0/1/5、seed02 pool3/2/6、seed03 pool4/7；每卡最多2条本run且全卡CUDA/pending少于2、空闲≥10000MiB才准入；全局1个CPU scorer，CPU2threads。资源池不会停止或热改其他任务。

第一处真实smoke/协议/来源/缓存/非有限/阶段技术失败停止新launch，健康已启动子任务保留完成，所有部分输出和其他owner保留；不自动重试。预期source_U_cache、smoke.json、9份source/segment50日志/默认头权重/complete、每行readouts预测/独立scores、逐行partial分析与三seed精确联合验收、pipeline状态与Git报告。训练完成不等于目标完成。
