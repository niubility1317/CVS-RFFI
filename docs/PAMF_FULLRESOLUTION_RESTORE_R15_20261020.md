# R15：不下采样 prior 的 TX 无标签恢复

完整目标保持：恢复器输入当前星地 residual/post-sync/noEQ received IQ，仅输出同一份 IQ 供原 PAMF 的全部派生分支；训练利用全部合法 L/U，训练不读取 TX 标签、F 权重或输出。正式 low urban≥70%，其余五场景各与同 F/seed 原始 raw clean 差距≤5pp，恢复后 clean 损伤单列。N607 默认、允许多 GPU，源规则固定后才可目标确认。

用户六类网络报告建议保留不下采样残差 TCN 控制。R14 的源反事实中，更准的已知接收作用使平均 NMSE 降15.4053%，但低城区仅+0.5556pp、中城区−0.0556pp，未通过固定身份门槛。R12 的额外解析统计损失也未通过身份验收。因此本轮检验一个独立假设：现有多尺度 U-Net prior 的池化/插值表示是否限制了输入细节保持；不假定更多物理层或更低平均误差必然保真。依据只取用户报告和源证据，不使用 R11 目标分组设计。

两行从零训练新 prior：`unet_control` 为原 Core(incoming=4)，`tcn_prior` 为实数 I/Q 通道全分辨率 TCN。TCN 以5点Conv4→128为stem，6个原Block(128,128)，dilation=1/2/4/8/16/32，保持每层长度256；末尾1×1输出2通道，zero-output初始化。原Block使用Conv5(膨胀)/Conv3、GroupNorm4、SiLU和固定残差缩放，不使用下采样/插值、BatchNorm持久更新或新损失。卷积名义感受野269，GroupNorm还作用于当前样本全窗；不声称严格复数相位等变。U-Net与TCN都是零输出头，因此完整恢复器初始输出相同，但隐藏参数形状不同，不称为相同随机初始权重。

两行都严格加载原R07末轮完整coarse/action，冻结参数与模式；没有加载R10/R12已训练prior。父coarse20+action20、新prior20=组件累计E60。原F50历史E50/11100另列、默认头不改。加载父网络后重新设置model seed2026092701，再构造新prior，避开R10旧初始化顺序偏差；新control不是原R10权重逐值复现。

相同6300L/56700U、固定20轮/4940更新、batch256、reset AdamW、2e-4预热247步后cosine至2e-5、WD1e-4、clip1、FP32。数据顺序seed2026102001；复用原R07的20轮缓存（15%clean/85%residual），不是新噪声或新物理样本。每行L126000/U1134000曝光。损失仅配对IQ NMSE+0.1 FFT logpower L1；没有标签/F教师、RMS校准、解析统计、场景/TX/RX专属分支。仅prior更新，原噪声估计与MAP固定6CG、原SCALE²，输出IQ不裁剪。

固定最后轮后，每行自动完成原1800sourceV七视图预测，再由独立CPU接V标签评分。V不更新模型。source身份验收是标签辅助选择，恢复训练0TX/F不能解释为整个研究选择完全无标签。只有TCN主候选可晋级，原R10native作固定源参考：平均六场景NMSE≤1.05×新control且≤1.05×原native，clean NMSE≤0.02；low/mid urban accuracy各至少比两者+0.5pp；clean相对原raw输入下降≤0.1pp；最差低城区TX相对control及native均≥−1pp；其他五场景最差变化相对control及native均≥−0.5pp。不是用目标分数调门槛。

全部通过后，同一owner自动对两行执行原始168000物理记录×七视图正式确认，原F50/defaulthead冻结；每行所有七预测固定后才独立truth-last评分，报告三郊区/三城区/clean及TX/RX/day、repair/damage、NMSE/FFT/配对特征与成本。失败保留两行源结果、0target，不救跑control。正式五场景下限依据原raw clean84.6761904762%，即79.6761904762%，不能降低恢复后clean来满足5pp。该目标基准此前暴露，不称新独立盲测。

单一不可覆盖run/release、N607普通账户、GPU池0/1/5/7择空卡、最多两条本任务worker、每卡一条，所有owner cap2/free≥12GB；capacity4共享锁保持到CUDA PID可见，子进程不重复锁。入口先实际2Ls/真实父权重no-query smoke，同时验证两结构初始完整输出相同、原F可读取、梯度只达新prior；PASS直接继续。技术故障停失败行、暂停未启动行，健康任务继续，不自动重试或性能止损。

|ID|来源|要求|目标文件|状态|验证|说明|
|---|---|---|---|---|---|---|
|A15-01|报告TCN控制|全256点表示、复用原Block、近参数量、单IQ|models.py|verified|长度/初始输出/梯度/逐样本测试通过；TCN793986、control834882参数|并未声称已改善TX|
|A15-02|无监督优先|全L/U、0TX/F训练、同预算/观察/顺序|training.py/design.py|verified|实际两行20/4940，L126000/U1134000曝光；0TX/F/target/冻结父更新|不把V作为训练|
|A15-03|继承约束|只继承R02/R07，无R10prior，seed加载父后重置|runtime.py|verified|严格来源负测与状态往返；实际2Ls初始完整输出相同、仅prior有梯度|新control不冒充旧R10复现|
|A15-04|冻结F有效性|原F50/defaulthead，七预测先于独立truth|evaluation.py|verified|新loader全七视图接口、F版本计数器及角色负测通过|训练不打开F|
|A15-05|源筛选|固定TCN主候选阈值、control和历史native护栏|evaluation.py/design.py|verified|身份/NMSE/最差TX双护栏、clean/其他场景退化及来源负测通过|条件性自动target|
|A15-06|执行保护|唯一owner/共享容量/不可覆盖/故障保留|controller.py/runner.py|verified|一次P0P1及N607 preflight/归档/compile；GPU5/7 PID/CWD/参数/loggrowth核实，全部自然退出|无额外门槛|
|A15-07|完整报告|三郊区三城区clean、TX/RX/day、成本、正式70%/5pp|本run报告/汇总|verified|实际源七视图及分层完整；源门槛未过，0target，正式70%/5pp尚未完成|源结果不冒充正式结果|
|A15-08|外部有效性|跨seed/独立物理族/逐细节真实|后续确认|deferred|单seed首轮源证伪，须通过再扩展|保持完整持续目标|

实际结论：SOURCE_NOT_QUALIFIED；源低城区 native / 新U-Net / TCN 为 82.1667% / 82.5000% / 78.4444%。详见[完整报告](../automation_reports/CV-SincNet/20261020-phase1-restoration-fullresolution-selfsup-s2026092701-m2-r15/report.md)。不改阈值或选择性重跑，目标保持 active。
