# R29：三步小波 U-Net 的高频特征保留对照

持续目标不变：单条当前 received IQ → 单条恢复 IQ，原 PAMF 从同一 IQ 派生 FFT、PA、时域和统计输入；恢复训练利用合法 L6300/U56700，0 TX 标签、0 已训练 F 权重/输出、0 target。允许模拟配对 clean。原 F50 seed2026092701/defaulthead 冻结推理；正式 low urban≥70%，其他五受损场景分别距原始 clean84.67619047619047%≤5 个百分点，即各≥79.67619047619047%。恢复 clean 损伤单列；不能把源域或波形改善当成逐细节 TX 归属证明。

问题来自 R28 源诊断和原模型实现：已知条件作用＋模拟器噪声代理使三步模型 mean6 NMSE 降低14.5600%，源 low urban 仅+0.4444个百分点，未达到原作用估计优先门槛。原 Core 每层 average-pool 缩减长度，奇偶差分不能沿下采样路径完整传递，尽管全分辨率跳连仍保留。R15/R24 全长 TCN/attention、R19 交替 FFT、R12/R22 解析辅助损失已经做过，不能把本轮写成首次时频研究。本轮检验一个具体尚未验证的机制：小波高频特征是否能在多尺度路径中增加输入细节保留。R27 目标分数不参与本轮结构、父来源或门槛选择。

方法依据为 [MWCNN 原论文](https://openaccess.thecvf.com/content_cvpr_2018_workshops/w13/html/Liu_Multi-Level_Wavelet-CNN_for_CVPR_2018_paper.html)：在图像恢复的 U-Net 中使用小波分析/合成。本轮是新的 1D IQ 假设，不复用图像预训练权重，不声称已有 RFFI 收益。正交 Haar 对每个特征通道分别计算 L=(even+odd)/sqrt2、H=(even−odd)/sqrt2，沿通道排列 L/H；逆变换严格交错恢复原奇偶位置。变换本身无损且能量保持，含卷积/非线性/通道压缩的整体网络不是可逆网络。特征高频也不等于已经识别的 TX 专属频带。

|行|下采样分析系数|其余结构|用途|
|---|---|---|---|
|lowpass_control|保留 L；H 置零|相同三步小波 U-Net、合成/跳连/参数/完整初值|匹配源对照；不晋级 target|
|haar_allbands|完整 L/H|与对照完全相同|唯一待资格判定的主候选|

每个 prior 为 incoming9、width32/64/96/128、三级长度256/128/64/32；每级两个原 Block（同包 GroupNorm4/SiLU）。下采样保留两子带，合成前 Conv1x1 映射至2×skipwidth，再逆小波上采样，与原同长度 skip 拼接后两个 Block；末层2通道零初始化输出。无硬阈值、learned mask、场景路由、批内混合、额外 IQ/FFT 输出或独立分类头。control 也保留全分辨率 skip，所以对照不是完全删除所有细节。两行同完整初值、同参数量、同 initial deployed 输出，三个不共享 prior 由同一初值复制，但参数独立。

接收链与原 R17 一致：只继承完整冻结的 R07 coarse/作用/原生方差，coarse 锚定；每步输入当前 received、coarse、当前迭代值、归一化伴随创新量和 logvariance，共9通道；三个 prior，每步2次 warm-start CG，固定原 SCALE²，总6次 CG。新 prior 从零训练，不加载 R17/R19/R23/R24 的训练后 prior，也不改变 R07 的估计作用。实际父链为 R02 coarse E20＋R07 action E20＋本轮 prior E20=组件E60，原监督 F50 E50/11100单列。R10/R15/R16/R17/R19/R23只作为冻结源评分参考，不是初始化。

每行从头E20/4940，完整6300L/56700U，每行L126000/U1134000曝光；model2026092701、split392005、data2026102001、原缓存 augmentation2026101912。复用原R07全部20轮训练 received，15%clean/85%residual，不增抽噪声。batch256、AdamW2e-4 warm247后cosine至2e-5、WD1e-4、clip1、FP32、固定末轮。目标只有逐包 IQ NMSE，clean 停梯度。无额外配对量、解析尺度拟合、F或TX训练信号。新小波网络使用既定单seed小矩阵，不直接扩种子或依测试分数救跑。

每行训练后立即自动完成源1800×七视图预测，再由独立CPU连接V标签。两行源评分全部完成后，主候选必须同时满足原native/R15/R16/R17/R19/R23六组源参考，以及本轮匹配lowpass_control：mean6 NMSE≤各参考1.05倍、cleanNMSE≤.02、low/mid urban各≥+0.5个百分点、clean相对原输入下降≤.1个百分点、低urban最差TX下降≥−1个百分点、其他场景最差变化≥−.5个百分点。规则复用既有资格函数，不放宽历史门槛。只有haar_allbands有资格晋级；不合格则0target。源选择使用V标签，整套研发选择不能称完全无标签；恢复优化本身不读TX/F。

若主候选合格，立即自动原168000物理记录×clean/三郊区/三urban完整预测，全部有限、ID匹配并固定后独立truth-last评分。每场景accuracy/Macro-F1、TX/RX/day、repair/damage、IQ/FFT/前232后24点/配对clean特征和实际成本均报告；正式结果不回流选模/参数/扩种子。低通对照不合格或主候选不合格均保留负结果，不访问target。该基准已暴露，不称新盲测或在轨验证。

最小实现验证：Haar解析常量/交错信号、能量/逆变换/复IQ相位混合、只消除H的对照、完整初值/容量与零头部署一致、三个prior的有限梯度、冻结父无梯度、同包独立和状态不变、真实完整checkpoint往返、错误来源/预算/F/TX/target负测、所有七预测完成前truth隔离、匹配control不可绕过与不合格target拒绝。真实2Ls/完整R07/F50 no-query smoke是入口第一步，PASS立即继续。此前F上下文问题使用no_grad构造，原头与参数缓存不改。

N607普通账户，新不可覆盖run `20261020-phase1-restoration-wavelet-selfsup-s2026092701-m2-r29`，唯一launch owner；池0/1/5/7、最多两条本run GPU worker、每卡一条，计入全owner cap2/free≥12GB，共享capacity4锁至CUDA PID可见。R28已有退出证据，其他owner健康任务不修改。固定来源错误、输出冲突、非有限或无法合法预测时保留产物、停失败行/holdpending、无自动重试；其他健康行继续。精确commit/命令/环境/CWD/路径/seed/expected artifacts在登记。

|要求|落实|状态|
|---|---|---|
|无TX/F优化、全LU、固定source角色及父来源|原训练循环/严格全状态来源|verified|
|单IQ接口、原默认头、三步CG不变|原UnrolledRestorer部署图|verified|
|Haar只替换多尺度采样，匹配低通消融|独立WaveletCore及数学/梯度测试|verified|
|源七参考与固定主候选、target隔离|独立模块资格/预测/评分接口|verified|
|实际训练、源/条件正式测试、成本和Git交付|登记与真实运行|verified|
|多seed/独立物理族/每个细节TX唯一归属|当前单seed模拟证据不足|deferred|

REJECTED_EXTRA_GATE：通用额外设计/spec审批、全仓库回归和报告单独审查不属于项目八项最小流程；沿用持续研发授权、聚焦检查和一次独立P0/P1，不新增审批或数据hash链。

实际结果：SOURCE_NOT_QUALIFIED；low/mid urban相对匹配低通 +0.1667/-0.8333 个百分点；完整训练/源与条件正式评分完成、进程退出；总目标未宣称完成。[完整报告](../automation_reports/CV-SincNet/20261020-phase1-restoration-wavelet-selfsup-s2026092701-m2-r29/report.md)。
