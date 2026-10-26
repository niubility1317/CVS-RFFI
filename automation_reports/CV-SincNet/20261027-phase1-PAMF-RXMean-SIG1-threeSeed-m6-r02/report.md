# 训练期 RX 均值差软一致性：两范围三种子

状态：RUNNING_VERIFIED。训练进行中，尚无正式性能结果，原目标 active。

只在训练期使用全 U 已知 RX 的五组低阶混合响应均值，构造同包的有限频率作用。作用是目标风格均值减原 RX 均值，经固定强度 0.1、幅度 ±0.5 dB、相位 ±0.025 rad 有界渲染；不减去单包响应、不把风格索引冒充真实接收机、不宣称分离纯硬件。原始 IQ 推理与默认头保持，无均值 bank 或新推理参数。

原普通 PL 的 raw 当前模型 eval 硬预测、0.95 阈值、weight1、全批归一化 CE 保持。新损失对全部 U 比较相同基础视图在均值差作用前后的软预测。raw 基础复用 PL teacher，H 基础使用 F(H) 的 detached 软预测，避免强迫受损 H 拟合 raw 硬伪标签。教师是当前学生，无 EMA；新增前向均 eval，学生保留梯度。30 个新增 U epoch 中 raw/H 各 15，H 使用已完成合法固定三副本三场景 raw cache。

U 与 LU 的标量均为 (nU*KL_U+64*KL_L)/(nU+64)，U 范围 detach KL_L，LU 才向 L 传播新梯度。每步复用原 T1 的 raw/As 输入选择 64 个真实 L，同样计算 teacher/student，保持范围消融前向成本一致。原 fullL 768/步、As 两次 CE/KL0、RDrop、浅中层跨 RX 及 SIG1/defaulthead 联合训练继续执行。

每个 seed 先用自己的实际 E20 父模型做 12 个源训练批的零更新梯度校准，raw/H 交替各 6 批；两范围使用同一系数 min(1,0.1*median(primary_norm)/max_scope_median(aux_norm))、444 步线性 ramp。这约束新增梯度剂量，不保证泛化。检查默认头与 SIG1 非零梯度、L 范围隔离、全模型 863730 参数与模型状态不变，PASS 后自动继续。均值银行仅拟合一次，model_seed=null，无 F/G 权重、V 或目标。

固定 RXMean-U、RXMean-LU × 2026092701/02/03 六模型；每行独立复用同 seed 合法 scratch E20/4440/13L7LU，再 +10+20 至 E50/11100/13L37LU。新训练共 180 模型 epoch/39960 次更新；共享父模型成本单列，不从协方差优胜模型或 F50 分支。没有新 PL/Sup 对照训练或预测。固定源合同、物理 ID、角色及已有 VALIDATED_ONCE 数据复用，不重验。

所有六行均固定 final E50，完成后逐行自动 source V raw+3H 诊断、raw clean+六 practical、cross_rx126000/cross_both42000、168000 包 ×7 视图预测，再独立 truth-last。源/目标分数均不改变剩余种子、配方、权重、epoch 或重跑决定。完整矩阵、精确命令及路径以 experiment.json 为准。已有目标基准暴露如实保留，不能称新的独立盲测。

原 raw clean≥85.981746%、六景>72.096197%及严格超过匹配 PL/Sup、原声明分层/范围的目标不变。即使此次 F0 三种子改进，也不能自动代表完整原目标达成。

队列最多四条本 run GPU worker，每卡本 run 一条、所有 owner 合计 CUDA+pending 最多两条，空闲显存至少10000MiB；独立 scorer 总并发1。健康其他任务不改动。技术错误停止新提交并保留活动产物；科学负结果不触发停种子。发布普通账号，一份代码 commit、一份 archive SHA 传输比对、remote compile，实际权重 smoke 是启动第一步并自动继续。

本版为独立技术替代 r02，r01 全留且0optimizer/target。修复 native runtime 在冻结依赖激活前导入的问题；六项focused tests PASS。只复用r01已完成全U20统计，新增U IQ读取0。强度、loss、两scope、三seed、+30预算和固定正式测试不变。精确输出目录使用rx_mean_source_r02。

发布执行 commit fa758a113f943f35a9da423fb1ca5cd3ff6a071c，archive 单次传输校验 VERIFIED、remote compile PASS。三seed各12批实际父权重联合梯度校准全部通过，参数/持久buffer不变、0optimizer/V/target，三系数均为1.0；这只是预固定剂量规则的结果，不是性能证据。

四份训练worker身份、CWD/argv、实际CUDA卡及步日志增长独立核实，当前01/02的U/LU在GPU1/3/4/5，03两行排队。原fullL768、AsKL0、普通PL及新增损失在已观察步骤实际执行，全观察U作用非零。这里只验证启动前缀，完整训练和正式三seed测试仍未完成。详见launch_verification.json；完成后逐行自动clean+六景预测/独立评分，不修改其他owner。

完成后核算工具已经落实为 `experiments.pamf_u_history.rx_mean_results`：只读取已固定预测与已有独立评分，先要求22job／六候选行全完成，核实24组同seed物理ID配对，再汇总7056000次候选raw决策、12份含对照评分。按混淆计数合并cross_rx／cross_both的3:1人口，报告逐seed、均值/样本SD、同seed配对差、每景及TX/RX/day等分层，不重读truth/IQ/weights。原协方差分析的既有完整结果在参数化共用统计后逐字段完全一致，旧结果不改。

完整日志核验入口为 `experiments.pamf_u_history.rx_mean_audit`，冻结行逐段解析全部step/epoch与完整stdout；查每个尾批124、raw/H各15遍、各RX／20个风格对覆盖、444步ramp及段间PL计数重置。当前六行尚未冻结，不能把启动前缀当完整日志分析。统计工具4项、既有统计回归3项、曝光核验3项测试通过，源训练代码和远端执行commit不变。

既有三seed36批源校准的新增分析见[calibration_analysis.json](calibration_analysis.json)。按选定alpha在full ramp参考下，raw组辅助/主梯度比中位数约3.57%至5.64%，H组约7.18%至11.51%；最大单批16.55%。原系数规则约束总体中位尺度，并非逐批10%硬上限，实际ramp早期权重更低。U范围10/36批、LU范围9/36批的辅助梯度与原fullL＋PL主梯度余弦为负；这是初始源模型的欧氏梯度局部关系，不能当作AdamW实际方向、泛化损失因果解释或改配方依据。高置信伪标签能支持身份监督，但不能保证额外正则目标每步同向。没有新增F前向/权重/IQ/真值读取，健康训练保持。

完整run结束后按现有runtime执行一次 `rx_mean_audit --runtime ... --output .../completed_source_audit.json`，再用 `rx_mean_results collect --runtime ... --output .../frozen_results_bundle.json` 和 `rx_mean_results analyze --bundle ... --output .../results`。未完成时collector拒绝给出正式汇总；不据中途源指标选种子或停行。

E30节点已经实际核实：[first_segment_audit.json](first_segment_audit.json)。01/02的U与LU四行均完成E20→E30，每行全部2220条step及10条epoch记录已解析，共8880步／40模型epoch，并扫描完整当时stdout，无非有限值或fatal。每行56700个U各10遍，曝光567000，raw/H各283500，20个RX风格对全部使用、各RX曝光113400；实际PL接受率93.3605%至93.6333%。这只是源训练执行证据，不是伪标签真实准确率，也不是目标性能。

四行均已通过E30→E50实际checkpoint smoke，后段已观察187至280个连续step；方法U历史从2221延续、原生段内PL计数按新段重置，5raw/5H已完成状态没有丢失，后段日程也未从第一个新U epoch重启。此处完整核验的是已结束E30段，后段仅为当时已生成的全部可解析日志前缀，不冒充E50完整日志。03两行继续按既定容量排队，无新target访问、无训练重启或配方调整。
