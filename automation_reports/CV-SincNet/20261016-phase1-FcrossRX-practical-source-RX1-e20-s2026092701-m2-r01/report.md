# F＋跨 RX 同 TX 对比：真实 practical 配对源检验

状态 REGISTERED_NOT_RUN。当前唯一假设：在用户最新 F＋CROSS_RX 基座上，保留一半共同普通增强监督、另一半采用原 practical residual 的配对 CE/RDrop，能改善源 urban 而不明显损失 clean。新 A_g 尚未成立；本轮不复活失败 OJ24，不加课程、HQ、对齐或 SSL 新规则。

两行分别从零 E20=13L＋7U，4440 更新，原 joint160／default cosine30／NoExplicitCFO／F 三路及统一 Sinc 每步物理单位 clip 保持。每行四次 L 前向：C0 两个相同普通增强 IQ 的 RD 对；C1 第一个对完全相同，第二个对使用 practical(original IQ)。各对权重0.5，每个 RD 复用同 IQ 两次 dropout、原 CE(LS.01)＋对称 KL(alpha1)，无两视图间一致性。这里 ordinary-clean 是现有 F 普通增强后的共同训练视图，不能写成未扰动 raw；其0.5监督份额两行固定，完整实际工厂/概率/日程保留。As 直接接收普通增强前的原 IQ，避免叠加普通 AWGN/多径/PA 后再施加完整星地信道。

跨 RX 对比沿用固定提交4df7b94d0f的 time_down/t2.pool、同 TX 不同物理 RX 正对、异 TX 负对，温度0.2、最大权重0.1，E6启用/E15达到0.1。只复用第一对两次前向，第三/第四次没有特征钩子；默认头仅通过原 CE/RDrop 联合训练，无额外投影头/参数/前向。U、own EMA、旧门控/cos/strong_noise_std保持，没有 U TX 真值或新 U 增强。

As 用原实际导入 leo_practical residual/post_sync、25MHz/IQ256、EQ关闭即G=1、reference30/reflect及原质量适配/相位跟踪/IQ补偿。几何、遮挡、传播、噪声、状态与接收残差保留原依赖。均匀采 low_suburban/mid_urban/low_urban，没有状态拒绝或质量重采；RX/day/eq硬件按session稳定且不含TX。流按epoch/step/view/opaquephysicalID刷新，训练与验证分离；新增噪声SNR不称总SNR。记录实际模块__file__、全配置及TX×RX×scene/state覆盖，不改正式IQ/信道/SNR/边界。

首折seed2026092701/split392005，fitRX3/4/6/8 L5040/U45360hidden/seenV21600，heldRX1 L/U不参与更新，heldV5400仅FINAL。每行训练冻结后自动预测该heldsource clean与全部六practical，每场景5400；模型state不变、0拟合。两行共同命名验证流，保存逐样本默认头/30间隔及全部状态/质量/物理量，独立复算后处理。无targetloader/IQ/truth、无正式部署性能声明；只有一个seed和一个sourcefold。

唯一首轮比较 C1−C0：source urban三场景均值至少+3pp、low至少+5pp、urban宏F1不下降，held clean及seenV损失均不超过0.5pp，完整有限更新、全部实际L/U序列相同和训练覆盖有效。FAIL关闭这套固定paired-As配方，不扫权重/强度/scene/换seed/RX救援；PASS只支持另登记source确认，不自动扩target或Ag组合。suburban与自身clean不超过2pp仅列为探索目标，报告实际各场景差距，不由下降clean缩小差距。

精确command/env/CWD/paths/GPU/stop在experiment.json：每GPU含其他owner/preCUDA总进程最多3，剩余>=12000MiB，矩阵最多两个worker，CPU2/workers0。不可变root，一个owner；低分不作技术停止，失败保留产物、无自动重试，不停其他健康任务。worker首步真实own scratchCP no-query smoke，PASS立即继续。所有旧权重/EMA/optimizer继承为0，旧F/E50结果只作历史背景；本次初始206状态张量与F一致。新基线由用户在既有暴露基准后指定，不声称新独立盲测。

新增成本两模型各E20、合计40模型轮/8880updates/4546560个L网络包，U/EMA/nativeV/heldclean/64800个源practical及烟测前向分项实测，墙钟实测。SupCon额外网络前向0；算子CPU时间和预算用尽不代表收敛。预期产物含实际配置/模块路径、full20日志及4440步梯度/物理clip、5994条L/U trace、默认头预测、曲线/分层/成本及same-row分析，具体清单在登记中。

REJECTED_EXTRA_GATE：generic brainstorming 的逐级设计审批和旧optimizer authority/hash链不在当前项目最小白名单；不新增审批或额外签名。保留唯一限定P0/P1检查和实际代码/config Git提交。

## 发布前实际验证

LOCAL_VERIFIED，尚未发布。两组累计1024合成网络包、188合成practical包、2合成optimizer更新、10个协议负例，0真实IQ/CP。206初态张量与原F一致；实际E6接口验证SupCon仅第一对两次前向，额外两个图没有特征对齐；默认头直接辅助梯度为None、其CE梯度及三SIG/Sinc联合梯度有限非零，ownEMA未被合成训练改写。相同第一视图、As输入直接来自raw、RX session稳定、跨epoch刷新、train/validation流分离与物理RX正对验证通过。

仅含实际发布代码的隔离目录（不在工作树sys.path）导入全部runner/source/dispatch/model/channel/publish并核对真实nativeargs20/13/7成功，0网络/CP/IQ。一次限定P0/P1发现worker锁父目录不存在的P1，已把锁移至已建立BASE，并一次定向读回PASS；其余范围PASS。审查额外只读，0数据/网络/optimizer。注册VALID、compile及N607普通账户preflight通过。唯一真实scratchCP无查询smoke在worker首步执行，PASS立即训练。
