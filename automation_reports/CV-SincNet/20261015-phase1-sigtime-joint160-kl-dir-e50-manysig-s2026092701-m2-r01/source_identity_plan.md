# 冻结后源身份诊断

本分析尚未发布，只在两条新 E50 行训练、七视图预测和独立评分均完成后运行；不修改训练、checkpoint、正式IQ或任何现有输出。唯一输出root为原run/source_identity_mechanism，唯一release cvs_sigtime_joint160_identity_source_20261015_r01，owner沿原任务。源输入复用固定 TX×RX×day 平衡 L720，U不开、无query、无额外分类器/优化器。

对象为同seed的已核实 scratch E50 B0、C3(A)、OUTPUT_KL、JOINT_DIR 四模型及各自 checkpoint 自有 EMA。own_model 分别验证原来源/源角色、scratch无父、final50/11100和source冻结；不使用任何目标分数选模型。以固定 C3 clean 正确集合为共同集合，C3 不是新训练的初始化父模型。clean与全部六practical源验证场景使用同一物理样本及已有命名draw，eval关闭Dropout，不把随机删除误差归为信道效果。

读取实际默认头和部署 joint160，观察原 base/PA/reference/SIG 分支，不增减模块。方向诊断对比同一源样本 clean、ownEMA clean、satellite；同时计算所有720个源clean向量中不同TX的最近余弦，报告正确配对余弦减错误TX最近余弦。不拟合分类器，不用此距离产生正式预测或新分类头。

每场景/visibility/lock/新增噪声SNR区间报告原默认头准确率、归一化间隔、固定A-clean正确集合保持率，相对A satellite的修复数与新错数；两者必须使用同一固定集合。报告分支/最终特征跨样本方差、坐标最小方差、范数、原gain，避免仅看辅助loss下降。记录 model/EMA state逐值不变、所有requires_grad=false、grad=None、head-fit和optimizer更新0。该源特征比较不冒充移除损伤的因果归因；受控/总效应由原四行E50 source_mechanism另行提供。

预期前向23040（4模型×[720 clean+720 ownEMA clean+6×720 satellite]），实际墙钟/计数以结果为准，父/训练成本另列。普通N607账户，仅CPU，CUDA_VISIBLE_DEVICES为空、torch/OMP/MKL/OpenBLAS2线程。命令 `/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -u -m experiments.cvs_urban_identity50.mechanism`，CWD PROJECT/releases/cvs_sigtime_joint160_identity_source_20261015_r01。首个真实C3/EMA clean前向校验actualjoint后打印 no-query smoke并继续。角色/来源/非有限/状态变化/输出冲突失败保留原产物，不自动重试、不影响健康训练。

本地发布命令 `E:/type10-7/local_envs/ssr-gpu/python.exe -X utf8 -m experiments.cvs_urban_identity50.source_publish --output E:/type10-7/local_artifacts/cvs_joint160_identity_20261015/source_delivery_r01`；独立 `--inspect` 核实PID/CWD/argv/CPU日志或最终source-only结果。沿既有只读source发布通道，一份archive一次传输比对/编译预检，不另加数据重验或审批。该分析不改变已经通过一次独立P0/P1审查的训练/评分候选，未变代码不重复审查。

新增3项定点检查通过：实际冻结网络/默认头前向等值、state/grad/模块结构不变；正确/不同TX的距离语义和隐藏标签拒绝；固定集合上的修复/新错计数。命令 `E:/type10-7/local_envs/ssr-gpu/python.exe -X utf8 -m experiments.cvs_urban_identity50.mechanism_checks`。J11保留pending，真实结果回来前不宣称机制改善。

预期 source_identity_mechanism/results.json、sample_identity.jsonl.gz、progress.json；原始physical级输出仅保存在运行目录/local_artifacts，不提交Git。结果汇总和分层图由同一结果生成，不能编造尚未完成路径或跨seed区间。后续中间视图、局部SIG教学、容差目标和U feature目标均未在此启用。
