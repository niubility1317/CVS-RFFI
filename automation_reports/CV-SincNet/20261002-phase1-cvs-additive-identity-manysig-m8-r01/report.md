# CVS 加性坐标注入：正式源实验预登记

状态 PLANNED，尚未发布或证明性能提升。普通 CE、无训练增强、仅身份骨干，性能优先。

设计依据是完整源域坐标信息探针和40行冻结内部消融，未使用目标成绩。同步形状与频率坐标联合保留，以同等参数的加性方向注入替代乘性gain；相对频偏含 TX/RX 混淆，不能称唯一 TX 晶振参数或任意接收机不变。

[完整数学假设与源证据](../../../docs/CVS_ADDITIVE_IDENTITY_HYPOTHESIS_20261002.md)。

## 本次执行与默认收尾

两核心各四 seed 2026092701..04，从零训练，无 checkpoint 继承；coordinate_equivariant 203193 参数，coordinate_gauge 164865 参数。保留原 L6300/V27000/U56700 unused、物理划分、E200×50、batch128、AdamW2e−4/wd1e−4/cosine1e−6、固定完整 FP32。640 参数 conditioner 使用频率圆周坐标及相干度，加性注入身份特征；每分量界为 0.25 倍核心范数，总扰动范数界为 0.25√160≈3.1623 倍核心范数，与上一轮参数量相同。全部详细文本、步记录、epoch JSONL/CSV、实际 conditioner 梯度、全源90单元、物理干预及资源实测保留。

8 个新记录加 4 个旧残差源指标，固定四 seed mean(0.5V+0.5最差源RX)性能最高优先；精确并列后才比较成本。仅 E200 选模，物理误差不参与排序或停机。旧权重不加载，目标数据不进入源执行。

若新模型胜出，冻结后默认 20261002-phase1-cvs-additive-clean-manysig-m24-r01：4 个选中模型 clean 预测加20个已冻结控制，共24行，168000 query/6TX/7RX，独立 truth-last 评分。若旧残差胜出，保留已核实历史测试，未选新模型测试 N/A；不追加 LEO/SFT/新增类。测试结果不得回流调参、结构、选模或选择性重跑。

每 GPU 最多两个训练任务，只使用空闲容量；一个 launch owner。不热改、停止或重启健康任务；技术失败保留产物，不自动重发。

46 项针对性验证及一次独立 P0/P1 审查 PASS。[验证](evidence/local_validation.json) · [实时源控制与输出不存在核实](evidence/source_control_preflight.json)。
