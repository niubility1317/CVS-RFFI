# 时序自挑战：02/03 条件扩展预登记

状态：LOCAL_VERIFIED／PREREGISTERED／NOT_LAUNCHED。原首种子仍在训练，完整源资格尚未产生。此报告不代表源资格通过或性能改善。

代码 `2f99d65f8a594ab1f56d84d1d7ee1011740da52c`，运行 `20261015-phase1-temporal-rsc-oldgate-cos-e50-clean-manysig-s0203-m8-r01`。只复用既有时序自挑战公式和训练框架，新增两个预定义模型种子 2026092702、2026092703，不重训 01，不改变或覆盖已发布任务。原首种子为 `20261015-phase1-temporal-rsc-oldgate-cos-e50-clean-manysig-s01-m4-r02`。

每种子三新臂 SIG_TIME、MATCHED_RANDOM、TEMPORAL_RSC，另复用同种子的历史 OLD_GATE_COS50；共 8 row、6 新模型、2 历史模型。每新模型 scratch50、33 label＋17 pseudo、11100 更新，固定 SIG_TIME195041／默认 cosine6×160 联合头／NoExplicitCFO、原 batch_neighbor＋ownEMA＋cos、U 真值隐藏和原数据角色。相同三 L 前向成本，6 模型合计 199800 个 L 网络前向，其中 133200 个额外前向，累计 300 新源 epoch／66600 优化更新；U 与教师的原前向保持。

所有机制字段、mask seed、两个 2/3 分位数规则、原 U 损失和日程与首种子相同。六个实际原生上下文仅 seed、run_id、candidate_id、output_dir 不同；同 seed 三臂初态一致。实际前向／L loss 接入、t1／t2／SIG／普通 W 梯度、原 U loss 对象、临时遮挡清理及 12 个负例已通过。唯一一次 adapter-only P0/P1 审查通过，没有重审未改的训练机制。

条件发布：只有原首种子完整源资格按已发布规则 PASS 才能启动三臂全部 02/03。主代理先提交并推送源域决定，再回收新模型目标数值；不得根据目标成绩扩展、取消、选父或重跑。`dispatch`、`source` 和远端 `preflight` 复用原源资格检查。没有另设科学阈值，也没有新增数据验证、签名或发布审批。

每 row 固定 final50 冻结，自动完成 clean168000 默认头预测，之后独立 truth-last 评分；原分类头随骨干联合学习，不专门拟合。保留全部 seed 与正负结果。仅两个新增 seed 的报告不能替代包含原 01 的完整三种子 84% 目标。完整实际部署特征／W 边界检查、额外前向、保存梯度与遮挡、原生 step 和完整训练曲线均复用首种子实现独立重算。

普通 N607 现有 CVS-RFFI／FP32／每 worker 两个 CPU 线程。入口 `python -u -m experiments.cvs_temporal_rsc_seeds.dispatch`；CWD `PROJECT/releases/cvs_temporal_rsc_seeds_20261015_r01`，不可变输出 `PROJECT/runs/20261015-phase1-temporal-rsc-oldgate-cos-e50-clean-manysig-s0203-m8-r01`，owner `codex/root/temporal-rsc-seeds-20261015`。共享容量计入所有 owner，每 GPU 最多 3 个实验进程，发布时至少 12000 MiB 空闲。发布时一归档一次 SHA 对比、远端 compile、真实固定首 SIG_TIME checkpoint 无查询 smoke；smoke 后直接执行。

仅停止 run 所属的系统性技术失败：数据／隐藏 U／默认头契约不符、非有限损失或梯度、固定遮挡不可辨认、随机回放失败、预算缺失、重复启动失败或输出碰撞。保留部分产物，不自动重试，不触碰其他健康任务，不依据目标停止。归档包含已提交的 experiment.json、report.md、events.jsonl；完整 canonical 配置仅替换 PROJECT 路径前缀，不折叠 source_config，读取时仅登记文件还原该前缀。

预期产物：每 row 的原生配置、源契约、初始化／final50／freeze、11100 step 与遮挡计数、4 个梯度时期、实际 feature-only 导出证据、prediction／独立 scores／completion；矩阵 source_features、source_observations、analysis 和 completion。原始 IQ、权重、PT、预测、日志与私密运行信息不入 Git；实现、登记及聚合正负结果提交推送。
