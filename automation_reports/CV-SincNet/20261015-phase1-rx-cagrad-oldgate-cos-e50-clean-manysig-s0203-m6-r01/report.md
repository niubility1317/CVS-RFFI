# 源 RX 梯度协调：固定剩余两个种子

状态：PREREGISTERED。实现提交 525dd1fe619b9524a2fce6188494bf7984e23171；源资格PASS及原样扩种子决定已在 dda55abb4cd793d65f5e63341d3747c762b6d14e 上传，在读取首轮新目标汇总前冻结。

固定矩阵：2026092702/03 × OLD_GATE_COS50、NORM_CONTROL、RX_CAGRAD，共6行。四个新模型各从零50轮，33L+17U、222step/epoch、11100更新、U963900；原SIG_TIME/joint160及195041个f_theta参数、默认cos头正常联合训练。两个旧门控+cos基线只读复用原同契约50轮产物，首seed不重训。新增200epochs/44400updates；与首seed合并新增300epochs/66600updates，历史3seed150epochs另计。

机制、系数c=.5、同范数自身状态控制、五RX样本数加权参考、SLSQP、失败回退和全部原生L/U训练设置均不变；不新增参数、模型前向或分类头，不单训/重训头。原U伪标签、batch_neighbor、ownEMA、lambda_U=.16/entropy=.01和cos保留。数据复用原VALIDATED_ONCE ManySig/split392005：L6300、U56700隐藏TX、V27000仅健康，源RX1/3/4/6/8 day1/2/3，10%标签；无新增星地增强，NoExplicitCFO且保留包内补偿。训练从零，无不明父权重；教师仅自身student。

源资格仅决定是否执行预定02/03；后续模型不按目标成绩改系数、排序、停止或选择重跑。固定final50默认头clean预测；每行全部预测固定后独立truth-last评分并自动报告。六行1008000决策/108条指标，与首seed合并9行1512000决策/162条指标。报告整体、MacroF1、每RX/day/TX、三seed样本标准差与配对差、错误修正/引入及实际缺项；84%只有三seed目标clean均值达到才可认定。已暴露基准，非新盲测。

执行：N607普通账户、现有CVS-RFFI原生FP32/CPU2线程；CWD=PROJECT/releases/cvs_rx_cagrad_seeds_20261015_r01；命令python -u -m experiments.cvs_rx_cagrad_seeds.dispatch；owner=codex/root/rx-cagrad-seeds-20261015；不可变输出PROJECT/runs/20261015-phase1-rx-cagrad-oldgate-cos-e50-clean-manysig-s0203-m6-r01。现有全局receiver_residual_capacity4锁，计入所有owner和启动保留，每GPU最多3个总进程且空闲>=12000MiB，不改变健康任务。发布入口experiments.cvs_rx_cagrad_seeds.publish --output LOCAL_ARTIFACTS/rx_cagrad_seeds_delivery_r01；仅发布一次。

检查：四个实际原生seed上下文及11个新增适配负例PASS，唯一新增适配直接P0/P1检查PASS；原核心未改，不重复全套审查。一次N607预检、实际源checkpoint no-query smoke、单归档SHA比对和远端编译后立即执行；启动后核对PID/CWD/argv/GPU/日志增长。

停止规则：仅本run数据/来源不符、非有限/原生跳过更新、非法U真值/目标/头操作、梯度公式/默认W校正/独立复算失败、输出冲突，保留产物、不自动重试、不停止无关任务。QP失败明确回退原CE；不以目标分数停止健康训练。

预期工件：preflight/launch/queue；每行resolved/初始化/50轮曲线/11100步日志/e1/17/33/50真实梯度私有向量/最终checkpoint/feature-only证明/固定prediction/独立scores/completion；全局source_features/source_qualification/analysis/completion及合并三seed分析。数据、权重、原始日志及预测不进Git，只上传实现、配置与聚合结果。源PASS及特征变化不证明目标泛化或新TX迁移。

实际发布：执行提交fd7cf6f1532082c38927bd632cc3ce19a486a2e9已独立核实远端，一次归档发布与远端编译完成，真实checkpoint无query检查PASS。四个原worker的PID/CWD/argv/GPU与日志增长已独立核实，当前各到第3轮，failure为空。历史两对照只读复用；保留原矩阵与所有健康任务，完成后自动逐行预测/独立评分。详见launch_evidence.json。

剩余02/03四个模型首轮实际梯度已独立复算VERIFIED：208层记录/832保存向量，selected等于原CE加校正且与保存目标一致；最大相对L2误差4.68e-7，默认W校正严格0，浅/中/SIG共156个采样均有非零校正。分析无模型前向/参数更新/头拟合/query读取；这是源接线证据，不是泛化结论。四个原PID继续运行，最新完整轮次13/14/14/14，未改变配方、预算或健康任务。

最终状态：ANALYZED/VERIFIED。固定三种子已全部完成，RX_CAGRAD 82.2885% ± 1.2216pp，对 NORM_CONTROL 的配对差 −0.293254 ± 0.502655pp、1/3 正；未达到84%。9行/1512000决策/162分层指标，完整审计和450源epoch曲线已归档。详情 [final_analysis.md](final_analysis.md)。历史启动和中期记录保留。
