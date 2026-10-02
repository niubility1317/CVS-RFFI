# CVS 相对滤波能量保留：正式源实验预登记

状态 PLANNED，尚未发布或证明性能提升。性能优先；普通 CE、无增强、仅身份骨干、原物理划分、clean-only。

[数学假设和完整执行矩阵](../../../docs/CVS_ENERGY_IDENTITY_HYPOTHESIS_20261002.md)。六个复数时间/行为 block 共用跨通道和时间的 RMS 分母，归一化本身保留滤波通道相对能量。后续学习尺度与径向门可改变比例；输入已有单位 RMS，未恢复绝对发射功率。received 能量包含 TX/信道/RX，不宣称唯一 TX PA 参数。

两个候选均 202553 参数：energy_equivariant 原输入精确直通，alpha=0；energy_half 全256点固定 alpha=0.5 波形校正。同一核心、深宽、参数结构，无新增学习参数、对齐标量或 conditioner。各四 seed2026092701..04，8个模型从零训练，无权重继承。

L6300/V27000/U56700unused、split392005、E200×50、batch128、AdamW2e−4/wd1e−4/cosine1e−6、完整FP32。实际配置/物理角色/来源按既有记录，目标不进入源执行。详细文本、step JSONL、epoch JSONL/CSV，实测CE/LR/全梯度/固定alpha前后和梯度N/A、V/最差RX、耗时/显存；记录实际六block归一化前后分数（每epoch末源batch和冻结公共干预）、90源单元与5TX×6RX公共干预。

8新源记录+4原残差源指标，四seed E200 mean(0.5V+0.5最差RX)最高优先，完全并列才比成本。物理误差只报告，不参与选模或停机。固定候选与源规则在新target访问前登记；全负结果保留。

新候选选中后默认 20261002-phase1-cvs-energy-clean-manysig-m24-r01：4新clean预测+20旧冻结控制，24行/168000query/6TX/7RX，独立truth-last评分。原残差胜出则保留已核实历史测试，未选新模型测试N/A，不重跑旧实验。目标评分不得回流结构/alpha/超参数/选模/选择性重跑；无LEO/SFT/新增类。

每GPU最多两训练任务且≥12GB空闲，仅使用剩余容量；唯一launch owner，不热改/停止/重启健康任务。59项定向源检查覆盖PASS；17项条件clean与17项前轮回归检查PASS；源与clean各一次独立P0/P1审查PASS；8个一次性CPU模型/24次CE验证PASS。未读取正式数据/历史权重/目标，不把合成检查称为识别提升。

[本地验证](evidence/local_validation.json) · [实时源控制与输出不存在](evidence/source_control_preflight.json) · [CPU检查](evidence/local_cpu_smoke.json)。
