# 原视图 AdamW 与分离状态：固定 E30 反证实验

唯一变化是自适应更新的组织。完整 practical、六次 L 前向、默认头、ordinary CROSS_RX、原 U 规则、物理数据角色和 E20＋E10 日程保持。MEASURE 两行执行同样两次 VJP 后保留 native AdamW；SPLIT 分别维护 rest/As 状态后以固定 0.8/0.2 合成一次自适应更新。原总损失不变，不能据此声称有效 As 更新份额不变。

不是 viewMoment 的单一 v 加分歧平方，不是 PCGrad，也不是连续调用两个优化器。Sinc 物理裁剪与衰减、一次成功更新和一次 EMA 保留。

三行固定 final30，自动 clean＋六 practical 的 prediction 完成后独立 truth-last 评分。源晋级只控制未来预算，不跳过负结果的正式评分。
