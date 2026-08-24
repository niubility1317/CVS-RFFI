# CAPTA-P0-LS Stage2-B设计规格

## 目标

在冻结的`ADV3B02_CORE90_SOFT_E200`编码器与CosFace头上，新增独立的`stage2_capta`路径。运行时仅使用`p2_min_v1`、`VALIDATED_ONCE`同row目标域received IQ、旧类support标签、冻结类原型/类别映射及预登记配置，形成一次性、support-only、零反向传播的目标原型迁移状态，并与`DA0_REG0`进行同row成对比较。

## 协议裁剪

附件给出的完整CAPTA设计跨越P0、P1和持续适配。本规格只实现当前协议允许、且能形成最小可证伪实验的`P2-LS`子集：

- 实现A0、A1、A2、A3和A6；主检候选为A3+A6。
- 冻结编码器、冻结CosFace头和冻结地面原型始终保留，训练参数数目为0，反向传播次数为0。
- A1以每类合法support中心和固定等效样本量进行球面贝叶斯收缩。
- A2以所有旧类support残差的类均衡均值估计共享目标域平移，再进行收缩。
- A3从同row有标签support相对冻结原型的类级残差矩阵估计至多rank-4正交子空间，仅保留共享平移在该子空间中的域码，再进行收缩。该实现是`p2_min_v1`兼容近似，不宣称使用报告中需要地面跨域原型学习的`U,V`。
- A6在query打开前，用support leave-one-out结果从固定混合系数集合`{0,0.25,0.5,0.75,1}`选择源/目标双路径权重；并列时优先更大的源路径权重。query只消费冻结系数和冻结目标原型，不更新任何状态。
- A4非均衡无标签软分配、unknown sink、A5 query图细化、地面方差/半径/跨域原型胶囊、P1梯度适配和CAPTA-C持续记忆均不进入本轮。前五项缺少当前`p2_min_v1`合法输入或会形成query更新/白名单外source统计；P1/C依照设计报告须等待P0稳定收益。

## 模块边界

- `code/cvsrffi/stage2_capta/prototype_transport.py`：输入冻结原型和support特征，输出不可变A1/A2/A3目标原型状态及资源审计。
- `code/cvsrffi/stage2_capta/safe_source_target_gate.py`：仅用support leave-one-out评分冻结双路径混合系数。
- `code/cvsrffi/stage2_capta/runtime.py`：校验Phase2上下文、冻结模型、提取support特征、构建状态，并逐query只读预测。
- `code/scripts/run_stage2_capta_p0.py`：复用现有已验证的row/package读取边界，提供`smoke`、`run-baseline`和`run-row`入口；`smoke`不接受query输入。
- `code/scripts/score_stage2_structured_late_block_pair.py`：继续作为同row`DA0_REG0/DA1_REG0`独立truth scorer，避免新增评分语义。

## 数据流

1. launcher核对`protocol_schema/capsule_id/split_id/phase2_data_status`并读取support-only包。
2. 冻结checkpoint提取support身份特征；任何模型参数或buffer变化均失败关闭。
3. 构建候选目标原型，利用support leave-one-out冻结源/目标门控。
4. 只有状态冻结后才打开query received IQ；逐样本面对全部旧类，输出源分数、目标分数、混合分数和prediction。
5. prediction完成后，独立scorer按opaque query token连接truth并生成旧类macro mean、floor及`DA1_REG0-DA0_REG0`。

## 失败与安全语义

- 非`p2_min_v1/VALIDATED_ONCE`、空句柄、非有限IQ、support标签不在冻结映射、checkpoint/原型不绑定、candidate/rank/shrinkage/gate网格未预登记时失败关闭。
- 禁止source/clean/cache/query truth/query role/query quota接口；禁止跨query分配和query状态更新。
- 输出不可覆盖；query前后模型state和CAPTA state必须逐值一致。
- 本轮科学晋级沿用旧目标阈值：旧类macro mean至少`+1.0pp`且floor至少`+0.5pp`。未达标记为科学失败，不把局部指标改善写成晋级。

## 验证范围

- RED→GREEN聚焦单测覆盖A1/A2/A3、门控、协议负测、零梯度/零模型变化、query只读与顺序无关。
- 邻近回归覆盖现有late-block row binding和paired scorer。
- 真实checkpoint无query smoke必须证明strict load、`source_input_count=0`、`query_input_count=0`、`backward_count=0`。
- 一次独立P0/P1审查后提交、push并回读远端OID。
- N607先运行单seed Target5同row最小矩阵；只有主检过门槛才扩展Target25。

