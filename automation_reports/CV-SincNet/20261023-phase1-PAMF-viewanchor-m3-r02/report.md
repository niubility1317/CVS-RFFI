# PAMF 物理视图特征一致性：三卡资源迁移

状态：RUNNING，真实 E30 checkpoint no-query smoke 已 PASS，三种子已分别在 GPU5/6/7 续训，尚无本版目标测试。此 run 仅替代 r01 尚未开始的 CPU 队列。资源读回显示原 GPU0/1/2 各有其他 owner 的三个计算进程；GPU5/6/7 各有约 23 GB 空闲显存和低计算占用。按用户授权仍只使用三卡，各固定一个种子，启动时总计算进程少于两个且空闲显存至少 10000 MiB。保留旧输出与其他任务。

科学方案完整继承 r01 在首轮目标分数读取前固定的假设，记录 commit 为 `5e52e4ebd383ba6990e9afd38a563798368f1603`。固定首轮 RXPSM 各自 E30/6660 的真实学生、自身 EMA、Gaussian 统计及 L-only 跨 RX 原型，追加 20 轮/4440 更新，累计 E50/11100、13L＋37LU。原 RXPSM E30→E50 是固定匹配对照，不读取或挑选其他 seed／epoch 权重。

只在物理强 U 子集新增 `0.10 * mean(w * (1 - cosine(z_student_physical, stopgrad(z_teacher_weak))))`，累计 E31 开始；普通噪声子集不接收新增项。沿用原 RXPSM、监督骨架、source-only Practical、默认联合头和 FP32/TF32off。没有额外推理参数或分类头。U_s TX 隐藏，训练配置只接受既有合法源数据句柄；V 仅冻结诊断，不据目标分数调参或重跑。

矩阵固定 VIEWANCHOR-s2026092701/02/03 → GPU5/6/7。精确代码 commit、发布命令、环境/CWD、数据/seed、独占远端 run/log/row 路径和预期产物以 experiment.json 的 code/execution/rows 为唯一维护位置。新 release 编译后只对本任务旧 CPU 控制器做 PID/CWD/命令、无子进程、三行全排队、无 smoke／row 产物核验；暂停后再次核验并确认退出，才启动新控制器。若旧 GPU 工作已开始则拒绝替换并恢复旧控制器，不重复派发。

21 项聚焦 SSL／继承／发布／资源检查 PASS。此前一次核心 P0/P1 审查 PASS；针对已发现资源阻塞的唯一一次定点复审无 P0/P1 发现。真实 E30 checkpoint no-query smoke 是新控制器首个 GPU 步骤，PASS 后立即继续三 seed 续训、冻结、七景预测及独立 truth-last 评分。两次相同系统技术失败停止新派发并保留健康任务；不按性能停止、重排或选择性重跑。

每行固定最终 E50 学生后预测 raw clean＋六种 practical，每视图 168000 物理样本，全部七景固定后独立评分。整体目标仍是同一方法对 F20、F21 的 clean 三 seed 均值分别至少 +1 pp，六景平均分别严格超过 +1.5 pp；整数正确数验收，逐 seed／景／RX／TX／day 和增量／累计成本均报告。当前无本版性能结论，不替换默认 PAMF 或 DURA。


## 发布和首次续训读回

实际执行 commit 为 `6226ad1e7cce97846702fe15ceba866d4acf64c9`。一次归档传输比较与远端编译完成；独立读回确认旧排队控制器已退出、旧 smoke 和 GPU worker 均未启动，旧 root 保留。新控制器及三 worker 的 PID/CWD/命令、实际 GPU5/6/7 和日志增长均核实。

真实 E30 学生／自身 EMA 的 source-only smoke PASS：累计 EMA 6660→6661，U 963900→964156，物理 U 481950→482078，新增特征损失 0.036681，L 前向包数 768，反向有限且 teacher 无梯度。smoke 只在内存测试，真实训练重新严格载入各自 E30/6660，并未继承 smoke 那一步。

首次完整 epoch 读回为累计 E32；当时三行已完整刷新的新增 step 前缀为 661／661／658 条，新增特征项均为正且有限，跨 RX 原型覆盖 1.0，U 和物理计数严格等于 E30 父数加本段实际曝光，没有归零。这是活跃训练前缀，不是完整 20 轮或目标性能证明。详见 source_activation.json；之后仍按 E50 固定学生和原七景独立评分收尾。
