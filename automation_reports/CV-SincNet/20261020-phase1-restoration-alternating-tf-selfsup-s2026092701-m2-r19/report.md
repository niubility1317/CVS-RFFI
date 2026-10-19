# 时域—频域交替恢复与同容量时域对照 R19

状态：**RUNNING / VERIFIED**。两行在 N607 GPU0/1 各完成首轮247次更新，PID/CWD/argv/GPU及日志增长已核实。方法 commit `5bbc0ccdc32de19e75cb6a01e2cda0352e39f36a` 已推送并独立核对远端 OID；34 项聚焦检查、compile/help、一次独立 P0/P1 无阻断和 N607 普通账户预检通过。真实 checkpoint 无 query 入口在第一步执行，PASS 立即继续。


目标保持：独立网络输出一份恢复 IQ，由原 PAMF 实际计算 FFT、PA、统计、时域分支并使用冻结原 F50/defaulthead。恢复训练使用合法全部 L/U 的模拟配对 clean，0 TX 标签、0 F 权重/输出、0 target。源 V 标签只作冻结后验收与固定排序。正式 low urban≥70%；其他五受损场景各距原始同 F/seed clean≤5 pp，恢复 clean 损伤单列。

本轮依据用户六类报告 R2-B 的时频交替 IQ 一致性，以及 R18 **源域**负结果：current_anchor/terminal_dc 虽提升 low urban 至 88.0556%/88.3333%，却使 TX20-19 从 R17 的 83.9590% 降至 79.5222%/77.1331%。这不支持放宽身份条件，也不证明某个 RF 分量的因果机制。R19 保留 R17 已通过源条件的 coarse 锚点与每步 2 次软 MAP，不继承 R18/R17 训练后 prior；不消费任何新旧目标分数来选择结构、损失或阈值。

两新行各三套不共享 Stage，每个 Stage 包含原 Core9 时间网络和新增 Core17 辅助网络（widths=24/48/72/96）。每行可训练参数3926220，总参数6083767；单个时间/辅助Core分别835842/472898。先由 received/coarse/current/单位 RMS 的真实伴随创新/lognativevariance 计算 `u=coarse+SCALE×temporal(inputs9)`。再根据 received、coarse、u、在 u 上重新计算的归一化伴随创新形成四组复数信号：

|行|辅助网络输入与输出|最终先验|
|---|---|---|
|time_time|四信号 IQ 8 通道、各自共轭 IQ 8 通道、logvariance 1 通道；输出 IQ 修正|u+SCALE×auxiliary|
|time_frequency|四信号正交归一化复 FFT 8 通道、正确负频率镜像共轭 8 通道、logvariance 1 通道；输出复频谱修正后 IFFT|u+SCALE×IFFT(auxiliary)|

频率镜像索引严格为 `(-k) mod 256`，不是同频点取共轭。两者随后执行 2 次 warm-start CG，初始值为上一步 current；共三 Stage、六次 Core 前向、六次 CG。所有可训练模块都置于 model.prior 内，必须进入同一个优化器。频域支路不独立送入 PAMF；推理输出仍只有一份 IQ。不输入真实 A/噪声/TX/场景，不做硬 Ax=r 或幅度剪裁。

两行新增辅助模块完整初值相同，每组三份从一份初始化克隆；原 temporal Core 完整初值与 R17 unshared epoch0 相同，只在真实入口比较，不加载旧 prior 初始化。新增辅助输出投影零初始化，因此两新行完整初始输出都须匹配 R17 epoch0。频域和时域表示后续数值路径不同，不宣称相同 FLOPs；参数和 Core 调用数受控，FFT、镜像与求解器成本实测披露。

各新 prior scratch E20/4940，全部 6300L/56700U，每行 L126000/U1134000 曝光；加载 R07 冻结上下文后重置 model2701，data2001，复用 R07 原20轮15%clean/85%residual 缓存。最终 IQ NMSE 唯一目标，不增加 FFT/解析/F 特征损失、标签监督、教师或校准。AdamW2e-4→2e-5、warm247、batch256、WD1e-4、clip1、FP32，固定最后20轮。组件 coarse20+action20+newprior20=E60，F50历史E50单列；三 Stage 不增加独立物理样本数。

每行固定 source1800V×七视图全部预测，再独立 CPU 评分。继续同时保护 native/R15FFT/R16IQ/R17unshared 四参考：mean6NMSE≤各参考1.05倍、cleanNMSE≤.02；low/midurban各≥+.5 pp、clean相对源raw下降≤.1 pp、最差低urbanTX≥各参考−1 pp、其他五场景最差变化≥各参考−.5 pp。只在合格者中按 low/mid 平均、lowurban最差TX、最差RX依次最高，最后名称字典序选一行。无合格者0target；有合格者自动完成其原168000×七视图、原F50/defaulthead和独立truth-last评分。R17既有正式对照只在新评分后复用，不重复控制预测。

N607池0/1/5/7，最多两条本run GPU worker，每卡一条；全owner cap2/free≥12GB，共享capacity4锁到CUDA PID可见。独立不可覆盖root/release、普通账户、唯一owner；真实2Ls/noquery入口PASS立即继续。技术失败保留输出、停止失败行/pending推进，健康任务继续，无自动重试；低成绩不触发技术停机。

本轮落实 R2-B 的一致时频交替思想；与 R3 的组合使用现有 warm-CG 接收链适配，并非报告梯度步公式逐式复现。整条选择及历史 F 并非完全无监督。单seed、同引擎代理、已暴露基准和逐细节身份真实性仍是边界。

|ID|来源|要求|目标文件|状态|验证|说明|
|---|---|---|---|---|---|---|
|A19-01|报告R2-B|时域/FFT/IFFT串接，只输出一致IQ|models.py|verified|真实FFT与归一化、最终目标梯度和逐包独立性本地检查通过|不独立输出FFT|
|A19-02|报告R2镜像|正确负频率共轭，复数尺度一致|models.py|verified|镜像恒等式/合成频谱修正/17通道检查通过|不作普通同频共轭|
|A19-03|容量归因|time_time匹配参数、6Core/6CG及完整初始输出|models.py/controller.py|verified|参数与初始化/完整初始输出本地匹配；真实权重入口待远端|非同FLOPs|
|A19-04|合法恢复训练|全L/U、仅R07冻结父、全部新模块进入优化器、0TX/F/target|runtime.py/training.py|pending|来源/schema负测及实际20/4940曝光、各子模块梯度|IQNMSE唯一目标|
|A19-05|身份保护|四源参考、固定排序、仅合格选定target|evaluation.py/runner.py|verified|未选target阻断、两角色实际七预测路由、原默认头及目标评分收尾负测通过|不降低旧条件|
|A19-06|安全发布|一次P0P1、真实checkpoint入口、Git/容量锁/N607poststate|controller.py|pending|聚焦检查和真实PID/CWD/argv/GPU/log|不可覆盖/保护其他owner|
|A19-07|完整证据|源全7、条件正式全7、TX/RX/day/损伤/成本、全日志|run报告|pending|独立truth-last、全量计数和汇总读回|源分数不替代目标|
|A19-08|真实性边界|新oracle/多seed/独立物理族/逐细节真假|后续确认|deferred|本轮不宣称、完整目标保持|非发布前阻断|

REJECTED_EXTRA_GATE：通用 brainstorming 技能的额外设计/计划多轮人工审批不在本项目最小流程白名单内。本轮已有用户“先尝试各种网络设计”及参考报告授权，保留实质设计与可核查对照，按项目最小流程继续，不增加审批或新的流程文件链。

发布前状态：34项聚焦检查通过，`python -X utf8 -m pytest tests/test_pamf_unrolled_restore.py tests/test_pamf_alternating_restore.py -q`；新模块 compile/help、Git diff检查通过。一次独立P0/P1审查无阻断，N607普通账户新root/合法缓存/源参考/实际容量预检通过。真实checkpoint/noquery入口、实际训练曝光和完整评分尚待执行。


完整矩阵、命令、环境/CWD、路径、GPU和停止规则见[登记](experiment.json)，[设计适配记录](../../../docs/PAMF_ALTERNATING_TF_RESTORE_R19_20261020.md)。


发布后状态：VERIFIED。单次归档传输 SHA 比对、远端编译、真实两条源样本检查通过。新时间权重与R17初值匹配，新两结构全初值相同，完整初始输出均匹配R17；全部六Core有梯度，梯度仅进入新 prior。控制器及两条 worker 的 PID/CWD/参数/GPU 和日志增长已核实；实际两张容量合规 GPU 各一条，未干扰其他 owner。无校准、无 TX/F/target 训练。逐行源评分及有条件正式确认继续自动执行。
