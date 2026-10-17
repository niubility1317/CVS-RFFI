# P35 有界时移 × 原 practical，E30 最小预登记

代码提交：7e4af64bb887c08c6a225bcb6bee4856fdb7c588。起点：d4391760b787aca67bdd298e26f2939e577becf9。单模型 seed 2026092701。

新增 B10=ordinary(x)/timing(raw)，B11=ordinary(x)/As(timing(raw))；复用 P32 B00/B01 原始 raw-secondary 同预算控制。各半 CE/RDrop 独立归一化后各 0.5，RDrop 复用同一已生成 IQ；原 SupCon 仅 primary，原 U/旧门控/ownEMA 不变。全源 L6300/U56700 隐藏 TX/V27000、全部 5 个源 RX，默认 cosine 头与原 F 网络联合训练。

每条新增训练从零 E30=20L+10U，222 步/轮，6660 次更新，3409920 个 L 网络前向样本。两条共 60 个新增模型 epoch/13320 更新；控制为既有 60 个历史 epoch，不能计为新增。预计新增总前向样本 13203856（训练、源读出、smoke 和正式预测均含）；以实际计数和墙钟为准。预算耗尽不等于收敛。

时移均匀整数 [-16,16]、reflect、不循环，恢复原始每包 RMS。训练 seed 202610174101，源诊断 seed 202610174102；私有命名随机流按物理观察与出现序号刷新，无 TX 标签依赖。As 原 residual/post_sync/EQoff(G=1)/ref30/25MHz/2.45GHz/reflect/质量补偿依赖不变；不对 As 输出额外 RMS。既有正式 IQ/信道/SNR 不变。

源依据 P34 只在 flat-H 固定状态反事实中支持延迟/边界贡献；相位解释失败，未证明本方法有效。kernel62/数值延迟24/参考80:160，在 ±16 下直接参考 IQ 依赖[3,176)，不声称所有神经分支避开边界。

源判据预先固定：B10、B11 全 V clean 各不低于 B00−0.25 个百分点；B10 两个独立 timing V draw 均超过 B00 至少 5 个百分点；B11 两 draw×两个原 full urban 均超过 B01 至少 1 个百分点。四项同时满足才获得本轮源证据。collector 先固定源判据，再读正式分数。失败即关闭当前固定方案，不扫描幅度/概率/窗口/RX/seed/epoch。失败也完成两条正式七环境评价，无自动扩种子。

每条 FINAL30 冻结后自动预测 clean+全部六种 practical，全七视图预测固定后独立 scorer truth-last。源资格每条16202NN，旧B00时移参考3602NN，兼容CP smoke2NN不作初始化；全部合法有标签卫星样本参与 CE。探索目标为 clean 相对 B00 正收益、low urban>60%、每种 suburban 距自身 clean≤2 个百分点，均非结果保证。

CWD：/home/szu2070436088/2510044040/CV-SincNet/releases/cvs_FcrossRX_timing_factorial_E30_20261017_r01

命令：/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -u /home/szu2070436088/2510044040/CV-SincNet/releases/cvs_FcrossRX_timing_factorial_E30_20261017_r01/experiments/cvs_timing_factorial35/runner.py

输出：/home/szu2070436088/2510044040/CV-SincNet/runs/20261017-phase1-FcrossRX-timing-factorial-e30-s2026092701-m2-r01；本地产物：E:/type10-7/local_artifacts/cvs_FcrossRX_timing_factorial_E30_20261017。GPU：共享锁分配、总实验进程/GPU≤3（含其他 owner）、空闲≥12000 MB；CPU2/FP32。单 launch owner /root。

停止规则：自身数据角色、来源、默认头、非有限值、功率/路由/更新预算或输出冲突等直接技术失败，保留自身日志及部分产物，不自动重试；不停止、修改或重启其他健康任务。科学 FAIL 仍完成已登记正式评价。预期保留配置、实际模块路径、覆盖/物理记录、训练曲线、冻结权重、源分类读出、七环境预测和独立评分、逐行逐 seed 结果与收益归因。

验证：7 项针对性测试和包编译 PASS；独立 P0/P1 检查 PASS（NaN 默认占位比较问题已修复并一次定向复核）。N607 普通账户路径/资源预检 VERIFIED。GitHub push FAILED（连接重置），本地提交保留。无新门槛或授权等待。
