# 固定 F 与 practical 实际调用：P43 后续审计

状态：VERIFIED_SYNTHETIC_ONLY。本轮没有新训练、源域准确率或正式测试成绩；联合提升目标仍 ACTIVE。P43 六行 E30 的源证伪和正式负结果保持，见 [完整结果](final_report.md)。

本轮代码：f65a8d94419acfe10972656864728565d2cde223。脚本为 [cvs_fixed_forward_audit_20261018.py](E:/type10-7/code/worktrees/phase1_urban_sigtime_20261014/analysis/cvs_fixed_forward_audit_20261018.py)，机器可读汇总见 [postmortem_audit.json](postmortem_audit.json)。P43 原训练代码仍为 19aedc3604fed9c193237dd92366164178af0983，没有用本次审计替换或重跑训练。

## 已核实与被否定的解释

1. 实际身份网络含 14 个 GroupNorm/LayerNorm 模块，没有 BatchNorm 或 running_mean/running_var/num_batches_tracked。合成前向及反向后 state_dict 完全不变。因此“clean/As 前向顺序污染 BN 运行统计”不适用于此模型；这不否定随机失活、RNG 顺序或优化轨迹的一般影响。不开展 BN 校准或换序训练。
2. 身份路径 232993 个参数均接入实际默认头的计算图。实际域骨干、域增强器、域头与对抗头在本次 forward_identity_only 中没有梯度；它们存在于完整模型，不能根据存在与否推断身份前向，也不据此修改固定架构。
3. ResidualPhysicalHead 的模块 forward hook 触发 0 次，但现有 W 有梯度；实际路径直接调用该对象的方法及函数运算。SequenceReference 的既有时序、参考响应、门控、PA 和融合路径均存在实际调用证据。不能用 cls_head 的模块 hook 沉默宣称头未执行，也不能用非零梯度宣称产生性能收益。Sinc 同样不能只靠模块 forward hook 判断是否使用。
4. 实际 leo_practical 来自旧固定 release 的 experiments/adv3b02_xuc/code/leo_practical，非根 code/。实际 Config 是 fs=25 MHz、fc=2.462 GHz、IQ256、reference_snr_db=30、residual/post_sync、EQ=false、reflect，质量适配、相位跟踪与 IQ 补偿启用。完整配置见 JSON。EQ 关闭时 G=1，没有“已均衡”含义，新增噪声 SNR 仍不是总 SNR。
5. 实际 RX 硬件 session 键为 RX×day×EQ，不含 TX 标签。合成双包改变 sample 坐标并从 E21/step4441 到 E22/step4663，硬件保持同一 session，信道 namespace 和输出刷新。没有向 apply 传 TX 真值。此检查验证训练 API 的机制，不是目标流随机性、所有源 session 的新统计或真实 RX 替换证据。P43 已有完整元数据重放另证实 all5RX、全部 L6300、810 个 TX×RX×day×scene×state 单元非空，本轮不重跑该覆盖分析。

## 旧负结果的边界

| 路线 | 实际证据 | 本轮决定 |
|---|---|---|
| 33tap、±0.5 dB、源平滑频谱 FIR | 冻结默认头身份保持基本通过；实际有限 IQ PSD 相对 RMSE 最大 0.601925，超过原 0.35。旧报告明确关闭固定核/幅度/control/阈值/seed/RX 救跑。 | 不以“旧诊断没有训练”为理由启动已关闭算子的首次训练。冻结输入失败不证明一切 FIR 训练无效，但也不允许改写旧条件。 |
| 共享正仿射 PSD | 432/432 拟合 a<0.5，主要向 RX 均值收缩；未保留预定 TX 关系优势。 | 不把谱误差下降当身份改善，不生成该模型的 IQ。 |
| ordinary/As 梯度投影 | 其他 owner 的三行完整 E30、七视图已 ANALYZED，低 urban 未达其固定增量。 | 不重复投影训练，不修改其健康/已完成产物。 |
| 不同 TX 实测 IQ 叠加且保留单一硬标签 | 没有本项目源身份依据；原 Mixup 对样本与标签同时插值。 | 本轮不启动此干扰增强。 |

旧源 FIR 与 PSD 分析使用历史 fit4RX/L5040；这是它们的证据范围，不是下一轮模型的训练合同。当前模型仍 all5RX、L6300/U56700 隐藏/V27000，无训练 RX 留出，E30，固定 F＋跨 RX 同 TX／默认头正常联合训练。

原论文对照：[Mixup](https://arxiv.org/abs/1710.09412)。这里只用于说明其标签定义，不将论文效果外推为本数据的物理标签保持证据。

旧证据：[FIR](E:/type10-7/automation_reports/CV-SincNet/20261016-diagnostic-F-crossRX-spectralIQ-fitL-s2026092701-m1-r01/report.md)、[PSD 正仿射](E:/type10-7/automation_reports/CV-SincNet/20261017-diagnostic-F-crossRX-PSDaffine-fitL-aggregate-m1-r01/report.md)、[viewPCGrad](E:/type10-7/automation_reports/CV-SincNet/20261018-phase1-F-crossRX-viewPCGrad-e20plus10-s2026092701-m3-r02/report.md)。

## 执行、成本与复现

已运行本地 AST 检查，以及普通账号 N607 的真实 release 合成检查。后者用 UTF-8 stdin 执行已提交脚本，设置 argv 等价于下列命令；没有新建训练 release、打开 IQ/checkpoint、访问目标或更新参数。脚本拒绝覆盖已有输出。提交后 GitHub compare 与本地 HEAD identical；远端完成 JSON 又经单独读取，未将退出码单独作为完成证据。

~~~text
python analysis/cvs_fixed_forward_audit_20261018.py --release <existing-P43-release> --output <new-audit-output.json>
~~~

已提交脚本这一次：2 个合成身份 NN、4 个合成信道 DSP 包、0 真实 IQ、0 checkpoint、0 优化/头拟合更新、0 teacher、0 U 真值、0 target，墙钟 1.964375 s。此前同轮的独立结构探测另有 2 个合成 NN；因此本轮新增合计 4 个合成身份 NN。前一个探测没有独立墙钟记录，不补造总墙钟。归一化与源码读取没有 NN。没有新模型 epoch、训练收益、源准确率、图表或跨 seed 证据。

原始 module 路径、完整 hook 调用列表及运行 JSON 保存在私有 local_artifacts 下；本报告只公布必要实现后缀与聚合，未提交真实 IQ、权重、原始日志或逐包预测。

## 下一项科学问题与停止条件

目前唯一保留的 Ag 问题是：是否存在可跨 TX/day 迁移、同时保留输入 TX 差异的共享、输入相关 RX 响应。引用对话正在开展对应的源统计，本分支不重复启动。它与已经失败的统一加性/正仿射收缩不同，也尚未证明可以实现为标签保持 IQ 增强。

最小下一步只复用该固定源结果，检查其跨 TX/day 预测是否优于 identity 与训练侧 RX 均值、是否保留正确 TX 对应关系，再决定一个有界 IQ 算子。任一核心条件失败就关闭该假设；不挑 RX、窗、方向、强度或 seed 救跑。统计通过仍不等于准确率改善。尚无通过上述科学检验并定稿的新训练候选，本轮未启动新模型。后续实际模型仍需保持 clean 监督份额，独立固定最小因子/匹配前向控制，按 E30/all5RX 训练并逐 row 自动七环境 truth-last，不能用现有目标分数选超参。
