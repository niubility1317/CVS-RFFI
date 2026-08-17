# Phase2报告中英文命名与ERTB-IDR纠正设计

## 目标

在现有Phase2详细报告中补充项目、Phase1和Phase2的正式英文名称，并将被简化写成`qKNN`的项目对比方法纠正为实际使用的D92 E0版本。修改仅作用于名称、首次定义和相关叙述，不改变实验数据、公式、参考文献或结论。

## 正式命名

- 项目中文名：星地跨域少样本持续射频指纹识别。
- 项目英文名：Ground-to-Satellite Cross-Domain Few-Shot Continual Radio-Frequency Fingerprint Identification。
- 项目简称：CVS-RFFI；简称作为既有项目标识保留，不在本报告中反向扩展字母含义。
- Phase1中文名：地面跨接收机域泛化射频指纹表征方法。
- Phase1英文名：Ground-Based Cross-Receiver Domain-Generalized Radio-Frequency Fingerprint Representation Learning。
- Phase2中文名：星载少样本域适应与新类增量注册方法。
- Phase2英文名：Spaceborne Few-Shot Domain Adaptation and New-Class Incremental Registration for RFFI。
- D92 E0中文名：高效稳健任务均衡增量判别注册方法。
- D92 E0英文名：Efficient Robust Task-Balanced Incremental Discriminant Registration。
- D92 E0简称：ERTB-IDR。

## D92 E0名称边界

`ERTB-IDR`只对应报告实际使用的`D92 E0_FULL_ONLY`版本，不代表原始D92。原始D92的正式名称仍为`RTB-IDR`。ERTB-IDR保留288维联合特征、ground-spectrum Cauchy稳健中心、旧/新任务均衡收缩协方差和全注册类统一LDA判别头；它关闭原D92的Fisher/Pareto安全门和K折full/block双几何融合，注册态只执行一次full主几何拟合。因此名称中的`Efficient`描述注册构造的显著简化，而不是新的性能晋级声明。

## 文档修改范围

1. 在第1章开头的红色加粗命名区，把项目、Phase1和Phase2名称改为中英文并列，并新增ERTB-IDR对比方法行。
2. 把报告中作为该对比方法名称出现的`qKNN`统一改为`ERTB-IDR`，包括Stage2-B、Stage2-C标题、正文和实验表格。
3. 重写首次方法定义，说明ERTB-IDR对应D92 E0而非原始D92，并简述输入、状态更新、判别几何和关闭的原D92组件。
4. 不新增ERTB-IDR论文引用；它是项目方法版本，不是外部论文方法。参考文献仍只保留现有对比论文。

## 验收条件

- 项目、Phase1、Phase2和ERTB-IDR的中英文全称均在首次出现处完整给出。
- `qKNN`、`ADV3B02`和独立的“原始D92结果”表述不再用于命名本次项目对比方法。
- 52张表、所有纯数值实验单元格、491个Word公式和5条参考文献保持不变。
- 全部页面完成Word渲染和逐页视觉检查，无截断、重叠、错位或异常分页。
