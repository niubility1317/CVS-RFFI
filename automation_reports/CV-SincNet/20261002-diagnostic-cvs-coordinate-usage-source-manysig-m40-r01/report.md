# CVS 坐标实际使用：完整冻结源域消融报告

8个合规E200模型×5种固定内部干预，40行每行27000个同物理源V样本、90单元；全部3600单元与40份混淆矩阵、Macro-F1、日志独立核对。8个trained行与原E200源准确率完全一致。没有优化、训练增强、模型状态更新、目标访问或源重排。

## 同权重内部干预

频率参考只置零两个圆周坐标；相干度参考只置零质量坐标（对应相干度1）。全部坐标参考保留学到的conditioner偏置；单位增益另移除输入响应与偏置固定重权。源波形及同步后的身份特征相同，仅修改固定权重的内部增益输入。这不是物理TX/RX干预或唯一硬件参数估计。Δ为干预减trained，负值表示屏蔽该部分后准确率下降，不作选模。

| 核心 | 干预 | 源V均值±seedSD（%） | Δ均值±SD（百分点） | 四seedΔ | 改变预测数均值/27000 |
|---|---|---:|---:|---|---:|
| coordinate_equivariant | trained | 97.3028±0.2411 | +0.0000±0.0000 | +0.0000, +0.0000, +0.0000, +0.0000 | 0.00 |
| coordinate_equivariant | frequency_reference | 97.3019±0.2432 | -0.0009±0.0056 | -0.0037, +0.0037, -0.0074, +0.0037 | 2.75 |
| coordinate_equivariant | quality_reference | 97.3028±0.2453 | +0.0000±0.0043 | +0.0037, +0.0037, -0.0037, -0.0037 | 2.00 |
| coordinate_equivariant | coordinate_reference | 97.3028±0.2456 | +0.0000±0.0052 | +0.0037, +0.0037, -0.0074, +0.0000 | 3.75 |
| coordinate_equivariant | unit_gain | 97.3222±0.2436 | +0.0194±0.0076 | +0.0111, +0.0296, +0.0185, +0.0185 | 17.25 |
| coordinate_gauge | trained | 91.6500±0.4032 | +0.0000±0.0000 | +0.0000, +0.0000, +0.0000, +0.0000 | 0.00 |
| coordinate_gauge | frequency_reference | 91.6278±0.4235 | -0.0222±0.0236 | -0.0370, -0.0407, -0.0222, +0.0111 | 41.25 |
| coordinate_gauge | quality_reference | 91.6398±0.4048 | -0.0102±0.0076 | -0.0111, -0.0111, -0.0185, +0.0000 | 14.50 |
| coordinate_gauge | coordinate_reference | 91.6278±0.4203 | -0.0222±0.0184 | -0.0296, -0.0444, -0.0111, -0.0037 | 46.50 |
| coordinate_gauge | unit_gain | 91.6046±0.4359 | -0.0454±0.0336 | -0.0704, -0.0778, -0.0111, -0.0222 | 89.00 |

完整逐seed CE/F1/变化数、每源RX效应及全部TX×RX×day单元均保存。内部依赖并不等于跨接收机因果身份信息，源V包含原5个源RX，不能称未知RX测试。该诊断不会改变本轮已经冻结的原残差选择，不宣称性能优化完成。

实际源权重commit `ecb268f7b4bcab111f041e177ecc3f967f1cb21a`；加载前核实完整源数据角色、scratch/E200及实际FP32，payload与source metadata逐字段相同且strict state加载。正式源IQ只在全部8份权重核实后读取；L/U没有迭代、V不拟合。每个模型全部干预后与加载时state逐张量完全一致，选择文件前后相同。一次矩阵owner按登记空闲GPU执行，耗时58.839s，峰值显存90482176bytes，硬件NVIDIA GeForce RTX 3090；这是全部流程耗时，不是星载训练或单次推理benchmark。

本次明确source-only诊断，不产生新clean预测/评分；不追加LEO、support、SFT、新类或D92三阶段。全部负结果保留，目标反馈禁令不变。

[40行结果](evidence/all40_rows.csv) · [3600源单元](evidence/all3600_source_cells.csv) · [全部源RX效应](evidence/all50_source_RX_effects.csv) · [独立核对](evidence/analysis_validation.json) · [实际远端读回](evidence/final_readback.json)。
