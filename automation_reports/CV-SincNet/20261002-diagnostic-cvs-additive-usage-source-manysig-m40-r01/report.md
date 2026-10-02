# CVS 加性坐标实际使用：完整冻结源域消融报告

8个合规E200模型×5种固定内部干预，40行每行27000个同物理源V样本、90单元；全部3600单元与40份混淆矩阵、Macro-F1、日志独立核对。8个trained行与原E200源准确率完全一致。没有优化、训练增强、模型状态更新、目标访问或源重排。

## 同权重内部干预

频率参考只置零两个圆周坐标；相干度参考只置零质量坐标（对应相干度1）。全部坐标参考保留学到的conditioner偏置；零加性项另移除输入响应与偏置方向注入。源波形及同步后的身份特征相同，仅修改固定权重的内部加性项输入。这不是物理TX/RX干预或唯一硬件参数估计。Δ为干预减trained，负值表示屏蔽该部分后准确率下降，不作选模。

| 核心 | 干预 | 源V均值±seedSD（%） | Δ均值±SD（百分点） | 四seedΔ | 改变预测数均值/27000 |
|---|---|---:|---:|---|---:|
| additive_equivariant | trained | 97.3333±0.2728 | +0.0000±0.0000 | +0.0000, +0.0000, +0.0000, +0.0000 | 0.00 |
| additive_equivariant | frequency_reference | 97.3056±0.2462 | -0.0278±0.0294 | -0.0630, -0.0407, +0.0000, -0.0074 | 70.50 |
| additive_equivariant | quality_reference | 97.3046±0.2762 | -0.0287±0.0082 | -0.0333, -0.0185, -0.0370, -0.0259 | 42.25 |
| additive_equivariant | coordinate_reference | 97.2713±0.2446 | -0.0620±0.0300 | -0.0963, -0.0778, -0.0407, -0.0333 | 84.25 |
| additive_equivariant | zero_delta | 97.2769±0.2417 | -0.0565±0.0344 | -0.0667, -0.1000, -0.0222, -0.0370 | 96.75 |
| additive_gauge | trained | 91.7694±0.4679 | +0.0000±0.0000 | +0.0000, +0.0000, +0.0000, +0.0000 | 0.00 |
| additive_gauge | frequency_reference | 91.5981±0.5046 | -0.1713±0.0745 | -0.1852, -0.2074, -0.0630, -0.2296 | 247.75 |
| additive_gauge | quality_reference | 91.6352±0.4629 | -0.1343±0.0407 | -0.0852, -0.1815, -0.1481, -0.1222 | 258.25 |
| additive_gauge | coordinate_reference | 91.4269±0.4967 | -0.3426±0.1119 | -0.2444, -0.4852, -0.2630, -0.3778 | 418.25 |
| additive_gauge | zero_delta | 91.7648±0.3735 | -0.0046±0.1809 | +0.2444, -0.0630, -0.0148, -0.1852 | 453.50 |

完整逐seed CE/F1/变化数、每源RX效应及全部TX×RX×day单元均保存。内部依赖并不等于跨接收机因果身份信息，源V包含原5个源RX，不能称未知RX测试。该诊断不会改变本轮已经冻结的源选择，不宣称性能优化完成。

实际源权重commit `d6c29b2bfbc5921a5ff654735253df21b1b1f709`；加载前核实完整源数据角色、scratch/E200及实际FP32，payload与source metadata逐字段相同且strict state加载。正式源IQ只在全部8份权重核实后读取；L/U没有迭代、V不拟合。每个模型全部干预后与加载时state逐张量完全一致，选择文件前后相同。一次矩阵owner按登记空闲GPU执行，耗时58.675s，峰值显存90482176bytes，硬件NVIDIA GeForce RTX 3090；这是全部流程耗时，不是星载训练或单次推理benchmark。

本次明确source-only诊断，不产生新clean预测/评分；不追加LEO、support、SFT、新类或D92三阶段。全部负结果保留，目标反馈禁令不变。

[40行结果](evidence/all40_rows.csv) · [3600源单元](evidence/all3600_source_cells.csv) · [全部源RX效应](evidence/all50_source_RX_effects.csv) · [独立核对](evidence/analysis_validation.json) · [实际远端读回](evidence/final_readback.json)。

## 下一步结构原型

40行源消融证明加性模型更多地利用坐标，但尚不足以超过原残差源分数。已形成波形层分数频偏校正原型：固定半校正与全类别共享一标量学习α，在现有等变核心前以exp(−jαωn)作用于全部波形；不增加层宽/深度/头或特征conditioner。14项本地数学/梯度/状态检查通过，原型没有正式训练、数据访问或测试收益。α不是TX/RX晶振占比估计，也不宣称整网频偏/任意RX不变。后续仍需新预登记、scratch和独立测试证明。[前瞻设计](../../../docs/CVS_FRACTIONAL_IDENTITY_HYPOTHESIS_20261002.md) · [本地检查](evidence/fractional_prototype_validation.json)。
