# F＋跨 RX 同 TX：真实 IQ 小比例组合的源身份保护检查

状态：PLANNED，算子和权限定向检查通过；尚未读取本轮真实源 IQ 或启动。

唯一假设：对同一 TX、同一日期的原始可见 L IQ，使用其他源 RX 的实际记录作轻度 donor，在保留 anchor 主要波形后，能否维持两个冻结默认头的身份判决。对照为 raw 与同 RX、同 TX 的另一包。它是观测空间的经验组合，不恢复、反演或替换真实接收机；同 TX 标签和 Mixup 形式本身不证明身份保持，更不证明域泛化。

依据：[WiSig 原文 IV-B](https://arxiv.org/pdf/2112.15363)说明 equalized 信号在处理后重新施加原频偏、重采样回 25 Msps 并取前 256 点；所用是已知 preamble。[Mixup 原文](https://arxiv.org/abs/1710.09412)只提供样本凸组合的统计学习形式，其非 RFFI 结果不能直接转用于本任务。本轮不根据历史或目标准确率确定幅度。

## 固定数据与算子

复用原 VALIDATED_ONCE source carrier，L6300 中先排除 RX1，保留 RX3/4/6/8 × day1/2/3 × 6TX ×原70包，即唯一 L5040。不是完整 L6300 训练，也不是 L5580/Ldev720。U/V 和 RX1 不参与本轮任何观察、默认头推理或 donor；原载体其他角色/特征虽反序列化，但未访问其标签/特征。两个冻结模型训练时使用过原合法隐藏 U 规则，本轮不使用 U 不表示父模型未用 U。

配对只用合法 L TX 标签和 RX/day 元数据：每个 TX/RX/day 单元按 opaque physical ID 排序，donor 为下一 rank 模70。跨 RX 对每个 anchor 遍历全部3个其他 RX，保持同 TX/day；同 RX 对照为本 RX 的下一包。不是同一次发射在多接收机上的时间同步记录，不把 donor 当作 anchor 的同一物理样本。没有类别、质量、置信度筛选；每个 anchor 均参与。

固定 lambda=0.1。先将 donor RMS 归一到 anchor，用既有参考的 lag20 和 [80,160) 规则估计 donor 相对 anchor 的频率/公共相位，只旋转 donor，再计算 0.9anchor+0.1donor，恢复 anchor RMS。anchor 的频偏不归零；NoExplicitCFO、必要内部补偿、完整 CVS 三路 SIG_TIME、统一 Sinc、joint160、默认 cosine30 头均不变。不做时移、切窗拼接、PA/量化/剪裁、额外噪声或传播，原 ordinary/As 不在本轮重复施加。全部256点使用同一组合规则，不制造拼接边界。

0.1 是固定保守工程范围，不是已识别的 RX 硬件幅度。归一化 donor 范数不超过 anchor 时，功率恢复后的相对 RMS 改变量上界为 2lambda/(1-2lambda)=0.25。新 view 因两包平均而降低噪声的可能性由同 RX 对照部分区分；只有后续 E20 真实 held-RX 识别收益才能说明泛化。

## 两冻结模型与成本

只读原 scratch E20、固定 final20 B00：seed2026092701（执行 commit 0ae066adb10a001e834c8c679c5b7b81c20b3ac8）和 seed2026092702（d71531d53c1d9ec46e3915963e05bff96760e80e）。其各自 immutable backend 和 source 路径在 experiment.json/实际 resolved config。精确核对 scratch/no ancestor、4440更新、final-only、default-head、未接触 target 及同一有效 L5040/U45360/V21600 契约。每个模型用自己原后端的独立进程加载，避免运行 profile 互相污染。不是新的训练初始化或追加训练。

CPU2、CUDA 空，两 worker 顺序运行。每模型 raw5040＋sameRX5040＋crossRX15120=25200 默认头推理；两模型50400，另仅 seed01 执行一次真实 CP 的2包合成 no-query smoke，共50402 NN。新 IQmix 生成40320个view，唯一源物理样本5040；不把虚拟 view 当作独立物理样本。0 optimizer/head update、0 teacher forward、0新训练 epoch。两个参照模型历史成本为40 model epochs／8880更新／9373356 NN，单独披露，未重跑或隐藏。

## 固定判据与停止

两模型均须满足：crossRX 相对 raw 的 overall accuracy、Macro-F1 各不下降超过0.5个百分点；每 RX、TX、day accuracy 不下降超过1个百分点；相对 sameRX accuracy 不下降超过0.5个百分点；raw 正确集合保留率≥99%；跨 RX view 相对 IQ RMS 改变量中位数≥0.01、最大≤0.25001；功率相对误差最大≤1e-5。逐 RX/TX/day、全部72单元和所有原始判决保留。以上是源 fit 身份保护和非空算子检查，不是 held-RX clean 或泛化成绩。

任一模型未通过，关闭固定 .1＋现有对齐组合方式，不修改系数、窗口、相位/频率对齐、RX或seed补救。全部通过也只支持另行固定和登记匹配 scratch E20 的三组：完整 baseline raw.75＋ordinary.25 保持，额外 .25 槽分别为 raw／sameRX mix／crossRX mix，8 L前向、原 U/EMA保持。未来 source 晋级要求跨 RX mix 相对两个对照 clean 各≥+1pp、clean F1不下降、各 practical/seenV≥-0.5pp；该训练尚未实现或启动，不由本 guard 的正结果代替。

技术 stop：缺输入、错来源/角色、输出碰撞、非有限、越界或 worker 异常，保留 partial/failure，停止启动后续本轮 worker，不自动重试；不修改其他任务。科学 FAIL 仍完成全部两个预定模型，以完整负结果报告。

## 发布和产物

owner codex/root/F-sameTX-IQmix-source-20261016；普通 N607 既有 CVS-RFFI Python；CWD PROJECT/releases/cvs_F_sameTX_IQmix_fitL_20261016_r01。命令 CVS-RFFI/bin/python -u -m experiments.cvs_same_tx_iq16.dispatch。独占 run root 为 PROJECT/runs/20261016-diagnostic-F-crossRX-sameTX-IQmix-fitL-manysig-m2-r01。单次 preflight、归档 SHA 本远端比较、remote compile、launch 后独立 PID/CWD/argv/CPU环境/log 或终态回读；无额外审批、封印/receipt链或数据重验。

本地21个输入/协议负例、5个科学判断反例、全部5040合成配对及 physical-order/TX/RX 置换检查通过，0真实IQ/NN/更新。初版逐坐标等变公差失败已定位为 float32输入量化，最大2.7e-6；float64参考最大1.15e-12。工程检查改为每包相对RMS≤8个float32 epsilon，实测最大5.90e-7；实际算子、科学幅度与身份判据未改。一次限定 P0/P1 独立审查 PASS，无直接阻断缺陷，启动 smoke 首次使用匹配真实CP，PASS即继续。

产物包括 root request/active/launch/result/completion/failure，各seed provenance/resolved、全部三view原默认头 logits/判决、私有 anchor/donor 物理ID及变换统计、完整日志。完成后独立复算全部50400源判决及保留率/分层/成本，生成图和报告、更新登记并提交推送。原IQ/权重/逐包标量/判决/物理ID/日志不入Git，仅保留聚合。没有 target 预测，truth-last target 评分不适用。A_g、A_s 独立收益、完整 B00/B10/B01/B11及交互、正式 clean＋六 practical 与跨种子联合目标仍未完成。

REJECTED_EXTRA_GATE：通用设计/计划审批超出项目最小流程白名单；本轮使用已有研发授权，不额外请求审批。
