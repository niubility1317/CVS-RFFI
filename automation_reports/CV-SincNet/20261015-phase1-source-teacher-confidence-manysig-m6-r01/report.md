# 固定 SIG_TIME 教师置信度次序：源域证伪预登记

状态 LOCAL_VERIFIED / NOT_LAUNCHED；代码 e2eee613b9657d336657012eb33d060072a402d7。本次不训练新网络、不改 SIG_TIME、不拟合或替换默认分类头，不访问目标/U。唯一独立 P0/P1 审查 PASS，真实 checkpoint 无 query smoke 在发布首步执行。

依据：完整同 L 强图诊断中，教师对学生错组的特征导数与身份 CE 在 18/18 相关对象同向；教师错学生对组 16/18 反向，但每模型只有 3 至 8 个 seen-L 样本。整体方向混合，不能认定错误教师解释了目标负增益，也不能用真值作门控。本次只检验一个可在 U 上计算的固定线索：FP32 默认头 softmax 最大值满足 teacher weak >= student strong 才保留；相等保留，差阈值固定 0，不扫温度、阈值、系数或种子。

矩阵 20261015-phase1-source-teacher-confidence-manysig-m6-r01：所有预定义 SIG_TIME ZERO / FEATURE_MT × model seed 2026092701/02/03，共 6 个已有 final50 学生及各自真实 EMA。父 run、完整 checkpoint 来源、数据契约和默认头见 experiment.json，加载前现有 source_proof 核对实际角色/物理 ID/33+17/11100/final50/EMA50，不从成绩最高的模型挑父权重。既有 split392005、L6300/U56700隐藏TX/V27000、RX1/3/4/6/8、day1/2/3、NoExplicitCFO 不变，复用 VALIDATED_ONCE。

合法 L6300：同一强图 trueCE smoothing .01 与原 unit joint160 半平方 L2、固定 .1 effective feature、.1 gated feature 及四对错组导数，全部按完整 batch N 作分母，不能除保留数放大；两臂都使用 .1 的诊断导数，ZERO 原训练系数仍为 0，不回放或冒充训练。对象只有实际 t1PW3456、t2PW6912、SIG projection576。gated feature 对比的分组 CE 仍是 full-group CE，明确不等同于只保留包的 CE。L batch256/末批156，shuffle2026101501、strong/noise/dropout2026101502。

V27000：固定 shuffle2026101503、strong/dropout2026101504，仅 no_grad 前向，梯度和拟合均为 0。记录教师/学生默认头预测、置信度、正确组保留率；规则本身不读取真值。资格规则事前固定：每个模型 V 的有益组（教师对、学生错）和有害组（教师错、学生对）均非空，且有益保留率严格大于有害保留率，全部 6 模型同时成立才 PASS；空组不可识别并 FAIL。任何失败均保留，不调阈值补救，不选择性重跑；PASS 也不自动发布训练。源 V 只证伪规则，不反向传播、不更新参数/EMA/默认头。

teacher eval/stopgrad，student 原 train dropout/strong noise .015；拒绝 BatchNorm，前后逐值比较全部参数与 buffer、不累积 .grad、不建 optimizer。U 的 IQ/标签/ID 0，目标 IQ/truth/统计 0；源 L/V 标签只用于各自获准诊断。本次源代理不能证明真实 U 准确率、训练效果、目标泛化、新 TX 或 Phase2 能力。

执行：普通 N607 既有 CVS-RFFI/fullFP32 CPU 串行六模型，Torch/OMP/MKL/OpenBLAS 各 2 线程，CUDA_VISIBLE_DEVICES 空，不占 GPU。CWD PROJECT/releases/cvs_source_teacher_confidence_20261015_r01；不可变 root PROJECT/runs/20261015-phase1-source-teacher-confidence-manysig-m6-r01；入口 python -u -m experiments.cvs_source_teacher_confidence.audit；本地发布 experiments.cvs_source_teacher_confidence.publish --output local_artifacts/source_teacher_confidence_delivery_r01。完整路径、环境/owner/命令均在 experiment.json。

预算：新增 epochs、optimizer/model/head 更新全部 0。每模型 L6300+V27000=33300 源样本观察、66600 forward 包次/262 forward batch、25 L 梯度 batch/75 层记录/1200 保存向量；六模型 199800 源观察、399600 forward 包次/1572 forward batch、150 L 梯度 batch/450 层记录/7200 向量。只有共享的 6300 L、27000 V 物理样本，不是六套独立数据。父模型仍各 50 epochs，历史成本不冒充本次新增训练。

本地验证 PASS：沿用 FCG 协议负例，另加 10 个负例，真实 SIG_TIME 合成 joint160/16 导数组、统一全批分母、全保留/空保留、ties kept、V no_grad、六模型资格及参数/buffer 不变；生成远端入口和 inspect compile PASS。唯一独立 P0/P1 PASS：未发现直接影响本次运行的缺陷；审查只读，无真实 IQ/模型/远端访问。发布只做一次普通账户 preflight、一个 release archive 的本地/远端 SHA 比对和 remote compile，首步真实 CP no-query smoke PASS 后直接继续；不加签名/receipt/重复审查。

技术停止规则：本 run 来源/数据权限/query/defaulthead/非有限数/状态更新/输出冲突失败则保存 partial、写 failure，仅停止本诊断，无自动重试，不触碰其他健康任务。源资格失败是科学负结果，不重启或调整。发布后独立核对 PID/CWD/argv/CPU 环境/loggrowth；完成后另进程重开源 observations 和向量复算全部统计，无新模型前向/反传。

预期产物：每 row resolved_config.json、source_policy_observations.pt、L_source_gradient_vectors.pt、analysis.json、source_gradient_recount.json、completion.json；全 run analysis/completion/preflight 与独立复算。原始 PT/日志留运行产物，Git 只交聚合 JSON、报告/图和登记，结果未生成前不宣称提升。
