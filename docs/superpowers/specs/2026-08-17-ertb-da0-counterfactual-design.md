# ERTB-IDR严格DA0_REG1反事实设计

## 目标

在现有D92 E0_FULL_ONLY Target125的相同125 outer、375 LEO scene row上增加`ERTB-IDR-DA0`反事实。该臂不读取target-old support更新backbone、中心、协方差、分类头或其他状态，只继承不可变Phase1旧类聚合知识，并使用target-new K-shot support完成新类注册。

## 状态定义

- `DA0_REG0`：原Phase1 bundle直接旧类预测；新类指标为`N/A`。
- `DA0_REG1`：旧类部分保持`DA0_REG0`状态；新类统计仅由target-new support生成；统一分类头面对全部注册类。
- `DA0_REG1-DA0_REG0`：无DA时的注册效应。
- 现有`DA1_REG0/DA1_REG1`保持不变；预适应效应分别报告`DA1_REG0-DA0_REG0`和`DA1_REG1-DA0_REG1`。

本臂是ERTB-IDR组件的因果反事实，不改名为主方法结果，也不把缺少target-old适应造成的下降解释为协议失败。

## 实现与证据

- 新文件`code/cvsrffi/stage2_d92_e0_da0_target125.py`冻结矩阵和method lock。
- 新文件`code/scripts/predict_d92_e0_da0_reg_only.py`只打开Phase1旧类聚合组件、new-support package和query package；target-old support descriptor必须保持未打开。
- 新文件`code/scripts/run_d92_e0_da0_target125.py`执行真实checkpoint smoke后完整Target125。
- 新文件`code/cvsrffi/stage2_d92_e0_da0_target125_analysis.py`执行同outer/scene的四状态与差分分析。
- 测试必须证明target-old support被打开、旧类状态发生注册期更新、query用于fit/update/selection、输出覆盖不完整或状态命名模糊时失败。

## 完整性

正式矩阵固定5 receiver×5 seed×5 slice=125 outer，每个outer固定3个LEO weak scene，共375个`DA0_REG1`scene row。技术停止只由协议/安全错误、覆盖写、错误checkout/hash或两个distinct outer出现同一pre-prediction确定性异常触发；不得按准确率停止。
