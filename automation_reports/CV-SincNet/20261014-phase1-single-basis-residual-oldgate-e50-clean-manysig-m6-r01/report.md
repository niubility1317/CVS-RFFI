# 单时间滤波基残差与相近参数量控制预登记

Run: 20261014-phase1-single-basis-residual-oldgate-e50-clean-manysig-m6-r01。状态：LOCAL_VERIFIED，尚未发布。代码/config 以发布归档内 release_commit.txt 的 Git 提交为执行版本；不使用报告自引用摘要。

本轮只检验一个假设：已完成 COMPACT_LINEAR 的多个时间滤波基，是否比相近参数量的单尺度残差控制更有价值。依据是合法源权重的 preGN 等效卷积核及多分支参与证据，以及用户明确要求补此对照。原头目标三种子 clean 均值 84% 保留，不承诺达标；目标成绩不进入配置、源资格、顺序、停止或选择性重跑。

## 矩阵与模型

固定模型种子 2026092701、2026092702、2026092703。6 个登记 row：3 个 BASE_REF 复用已冻结 OLD_GATE_COS50 的同种子默认头预测；3 个 SINGLE_BASIS 从零训练 50 轮、33 label＋17 pseudo、每轮 222 步，即新增 150 模型轮次、33300 次优化器更新。COMPACT_LINEAR 三个已完成历史 row 仅用于源健康控制和最终配对，既不作为初始化也不重训/重测。原基线与 COMPACT 的历史成本各 150 模型轮次、33300 次更新，单列。

SINGLE_BASIS 改 t1/t2/t3：单 k5 depthwise→两个无中间激活的 1×1 pointwise（hidden90/127/148）＋独立 1×1 skip，随后保留原 GN、激活、池化、dropout。时间模块参数 81744，相比 COMPACT 的 81672 多 72；整个身份模型 237769 对 237697。最大局部卷积覆盖与 COMPACT 相同，t3 为 142 个原始样本；GN 的全局统计依赖保留。每输入 preGN 时间核秩上界为 2，COMPACT 为 4；这不是 joint160 特征秩。

两层线性 pointwise 的隐藏宽度足够表达原单 pointwise 函数族，但带来优化参数化差异。相近参数量不等于相同函数容量。不同时间结构的初始化函数也不相等，不能把最终差异全归为滤波基数量。新增模块 CPU fork 隔离初始化 seed＋48213，恢复全局 CPU/CUDA 随机流；其余原参数和默认头起点一致。

Sinc、频域、PA、参考响应、原输入策略及内部逐包补偿保持。特征真实导出为 id_feat_joint160；base_norm、pa_norm、gain 属于特征计算，只有具体类别 W 属于分类权重。默认 ResidualPhysicalHead 6×160、scale30 按原损失随骨干联合训练。单独头训练、换头、原型替代、RX/day 分类器和 Phase2 均不执行。

## 数据与训练

复用现有数据契约和物理 ID：源 RX1/3/4/6/8，实际 native day1/2/3；L_s6300、U_s56700（TX 真值隐藏）、V27000。训练池标注率 10%，V 不进梯度。ManySig 原256点输入、NoExplicitCFO，历史 batch_neighbor＋EMA＋普通 cosine；AdamW lr0.0002、weight_decay0.0001，cosine 到1e−6，label_smoothing0.01，配方与本次已核实 A33＋17 基线一致。无新增信道增强。所有待训练模块均进原优化器和 own EMA。固定第50轮，不按目标选模。

首种子最终 L CE≤0.5、源 V 平均准确率较同种子已冻结 COMPACT 降低≤1个百分点、最差源 RX 降低≤2个百分点，才放行其余两种子。资格依据仅源证据；首种子冻结后的自动预测/独立评分可与后续源训练并行，测试分数及评分完成状态不作科学放行条件。源机制 Observer 另加每模型5次完整 V 前向，共135000观测，及固定 L 小批记录；不额外访问 U 真值或目标。

## 发布与停止

本地 CWD：E:/type10-7/code/worktrees/report_physical_ssl_delivery_20261011。
远端 CWD：/home/szu2070436088/2510044040/CV-SincNet/releases/cvs_time_basis50_20261014_r01。
环境：/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python。
命令：python -u -m experiments.cvs_time_basis50.dispatch。
唯一 owner：codex/root/time-basis-20261014，N607 普通账户，GPU 由现有锁调度；每 GPU 含其他 owner 总实验进程≤3，空闲显存≥12000 MB。不修改健康其他任务。
唯一不可覆盖输出根：/home/szu2070436088/2510044040/CV-SincNet/runs/20261014-phase1-single-basis-residual-oldgate-e50-clean-manysig-m6-r01。
本地产物：local_artifacts/cvs_time_basis50_20261014_r01。

错误数据契约/权重来源、U 真值访问、碰撞、非有限值、无法启动、预测或评分技术失败按原规则停止新增派发，保留健康在跑行和已有产物，不自动重跑。科学门槛失败保留首种子训练和自动测试，记录缺种子，不冒充完整三种子矩阵。

一次独立 P0/P1 审查发现发布包漏 factorial 依赖；已修复并仅针对该问题复核 PASS。本地实际算子及输入梯度折叠误差≤3.56e−15，参数量、秩上界、原非时间参数/RNG、严格模型/teacher roundtrip、CE梯度→数值更新→z改变及 W 修改/删除导出不变均 PASS。新源资格负例证明使用 COMPACT 源参考，而不是 BASE 源成绩。合成 IQ 接线检查不作为识别性能证据。发布第一步进行一次真实原基线 checkpoint 无 query smoke，通过即继续。

## 自动收尾

仅 clean：固定168000个物理样本、7个未见 RX×4天×原6 TX。每 row 全部预测固定后，新的独立 CPU scorer 才读 truth。本轮6个默认头读出共1008000决策（新增504000、原基线复用504000）、108条混淆矩阵。全部新登记行完成评分后复用 COMPACT 原有504000决策、54条混淆矩阵做配对，不计作新增训练或新独立实验。

输出 source 初始/终轮权重、实际配置/角色/epoch与step日志、stage32/33/34/40/50源记录、默认 joint160 导出检查和真实初始→最终核/参数变化；prediction 固定数组与完成标记；逐 row scores；全矩阵 completion.json、analysis.json、final_analysis.md。报告逐 seed/均值/样本标准差/正差种子数、accuracy、Macro-F1、最差 RX/day、SINGLE−COMPACT 与 SINGLE−BASE。clean 结果不称卫星部署或新 TX 迁移证据。已有暴露基准如实说明，保留全部负结果。
