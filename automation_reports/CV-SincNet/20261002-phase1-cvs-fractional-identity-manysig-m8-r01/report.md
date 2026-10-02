# CVS 波形层分数频偏校正：正式源实验预登记

状态 PLANNED，尚未发布或证明性能提升。普通 CE、无训练增强、仅身份骨干，性能优先。

[数学假设与源证据](../../../docs/CVS_FRACTIONAL_IDENTITY_HYPOTHESIS_20261002.md)。已有加性源实验与40行内部消融提示后置坐标注入贡献有限；本轮把received相对频率直接保留在波形作用位置，不把混合频偏当唯一TX参数。

两个候选使用同一equivariant_memory核心：fractional_half固定alpha0.5，202553参数；fractional_learned共享一标量sigmoid，初始化0.5，202554参数。全部256点身份路径先执行exp(-jalphaomega n)，样点振幅保留，alphaomega+y可重建原波形。常相位性质保留；仅有效且不跨主值分支的部分波形频率协变，不声称整网仿射不变、任意RX/LTI不变或硬件参数恢复。alpha不是TX/RX贡献比例。

各四seed2026092701..04，8个模型从零训练，无权重继承。原L6300/V27000/U56700unused、split392005、E200×50、batch128、AdamW2e−4/wd1e−4/cosine1e−6、完整FP32。详细文本/每步JSONL/epochJSONLCSV记录实测CE、LR、全梯度、alpha前后与一标量梯度（固定N/A）、V/最差RX、耗时与显存；完整90源单元、公共5TX×6RX物理干预与成本均保留。

8新记录+4原残差源指标，四seedE200 mean(0.5V+0.5最差RX)最高优先，完全并列才比成本；物理误差只报告，不参与选模或停机。旧权重不加载，目标数据不进入源执行。

新候选选中后默认20261002-phase1-cvs-fractional-clean-manysig-m24-r01：4新clean预测+20旧冻结控制，统一24行/168000query/6TX/7RX，独立truth-last评分。若原残差胜出，保留已核实历史测试，未选新模型测试N/A，不重跑旧实验。测试结果不得回流alpha、结构、超参数、选模或选择性重跑；无LEO/SFT/新增类。

每GPU最多两个训练任务，仅使用空闲容量；唯一launch owner。不热改、停止或重启健康任务。55项针对性检查和一次独立P0/P1审查PASS；8个一次性CPU模型/24次CE验证PASS，未访问正式数据或权重。

[本地验证](evidence/local_validation.json) · [实时源控制/输出不存在](evidence/source_control_preflight.json) · [CPU smoke](evidence/local_cpu_smoke.json)。
