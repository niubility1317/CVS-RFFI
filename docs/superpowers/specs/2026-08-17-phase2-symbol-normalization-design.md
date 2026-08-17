# Phase2报告符号统一设计

## 目标

在不改变方法机制、实验数值、表格数据和参考文献的前提下，消除同义多符号与一符多义，使导师能够按“集合—样本—状态—指标—损失”的固定规则阅读全文。

## 统一规则

1. 发射机类别集合统一写作`\mathcal Y`；`Y_old`、`\mathcal C_old`等旧写法统一为`\mathcal Y_{\mathrm{old}}`。普通大写`N`仅表示类别数量，`c`仅表示类别索引。
2. 数据域统一使用直立文本下标`\mathrm{src}`与`\mathrm{tgt}`，避免`t`同时表示target域和增量session。增量session保留`t`；优化步改用`u`。
3. support、query和batch等集合使用花体字母；集合基数使用绝对值符号。单样本或标量不使用花体。
4. 所有分类概率统一为`\pi_\theta(c\mid x)`，不再在不同方法中混用`p_\theta`和`q_\theta`。
5. 类原型统一用粗体`\boldsymbol\mu_c`；概率不再使用`p`，以避免prototype与probability混淆。
6. few-shot任务保留`\tau`；MoPC-HR的softmax温度改为`T_{\mathrm{temp}}`。数值稳定常数使用`\varepsilon_0`；高斯扰动使用`\boldsymbol\xi_r`。
7. 点分类损失保留标准记号`\ell`；MoPC-HR参数组索引改为`j`、参数组总数改为`J`，避免与损失函数`\ell`和分类器`g_\theta`重叠。
8. 预测状态不再混用`(0)/(1)`与`pre/post`。报告首先定义`DA0_REG0`、`DA1_REG0`、`DA0_REG1`和`DA1_REG1`；具体结果表必须明确该方法经过的状态路径。
9. qKNN名称中的`K`属于算法名称，不等于报告的`K-shot`样本数；首次出现处明确区分。

## 内容边界

- 保留17张表、153行表格记录及全部实验数值。
- 保留现有5篇对比方法参考文献，不增删或改写条目。
- 不改变ProtoNet、MRIOR-SDA、DADDA-SDA、CSIL、MoPC-HR和qKNN的方法机制、损失权重、训练步数或结论。
- 仅修改符号、符号说明、公式中的对应记号及为消除歧义所需的短句。
- 中文继续使用宋体，英文、数字和公式周边文本继续使用Times New Roman。

## 验收标准

- 自动检查禁止符号组合和必需的新符号组合。
- 文档表格数、行数、数值token序列及参考文献段落与输入版一致。
- OMML公式对象数量不减少，文档无`U+FFFD`、可见LaTeX反斜杠或独立符号说明段落。
- 生成PDF/PNG并逐页检查公式、表格、分页和字体显示。
