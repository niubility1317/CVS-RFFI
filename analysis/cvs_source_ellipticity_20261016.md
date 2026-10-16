# 源 RX 椭圆性：新物理候选先做重复性证伪

当前独立 Ag 尚未通过识别验证；paired-As 五源 clean 保护失败、真实跨 RX partner、C4、真实长窗与 RX-DC 合成方向的负结果保留。本项独立问题是接收 IQ 的 AC 椭圆性是否存在跨 TX、跨 day 稳定的 RX 关联。它不是 DC 的窗口或坐标救援，也不是已失败的频率响应 bank。

只用当前 C0 留 RX1 合同的 fit RX3/4/6/8 原 L5040，六真 TX／三 day／每 cell70；先按源 L 物理 ID 与 RX 元数据过滤，再读取选定 IQ。held RX1 IQ、U/V 值、U TX 真值不进入统计，0 target；复用 VALIDATED_ONCE 源缓存，不加载任何模型 checkpoint。源缓存可能同时反序列化其他源角色容器，明确记载但不使用这些值。

每包固定完整 256 点，先在描述量内去复数均值，再取 `abs(mean(x_ac**2))/mean(abs(x_ac)**2)`。只用椭圆性大小，不扫描复杂方向、频率、窗口或其他坐标；去均值仅用于统计，不改变训练或部署 IQ。该量在共同标量增益、整包相位和加性常数下不变；现有 PA、多径、噪声等复合普通增强仍可能改变它，不能据此宣称旧增强完全不覆盖。广义线性 RX 响应可改变椭圆性，但 TX、调制和传播也可改变它，源统计不能识别硬件因果或证明标签保持。

先取 TX/RX/day cell 均值，再减同 TX/day 的四 RX 均值。18 折同时留一个 TX 和一个 day，只用其他5 TX×其他2 day 估计 RX 对比，预测留出 cell；零 RX 效应作平方欧氏风险参照。固定资格全部满足：零效应总风险非零、聚合风险降低至少10%、至少12/18折降低、完整 source RX 均值对差 p95 至少0.01（椭圆性绝对量的一百分点，避免仅统计可检出但极小的效应）。这些是源先验资格，不是准确率门槛。

FAIL 关闭这一固定 full256 椭圆性引导的 RX 合成方向，不改复数方向、窗口、RX/TX/day 子集、强度或阈值救援。PASS 只支持另登记有界广义线性接收 IQ 的 E20 源身份训练检验；不自动训练，不自动进入 As 或目标，不反演未知 TX 效应，不预告 clean 或 urban 增益。

CPU2 线程、CUDA隐藏，0模型／网络前向／checkpoint／头拟合／优化更新／星地生成，E20预算消耗0。一个不可变 root、一个 owner；技术错误保留输出不自动重试。当前 G0 不使用模型，checkpoint no-query smoke 不适用，不加载无关权重。精确 row、代码提交、命令、环境／CWD、路径、停止规则和预期产物见该 run experiment.json；一次聚焦负例、一次限定P0/P1、单archive本远比较、remote compile和独立执行读回。

实际结果：VERIFIED／ANALYZED，PASS_SOURCE_STATISTIC_ONLY。聚合预测风险降低27.486642%、12/18正折，RX椭圆性对差p95=0.032791289171；TX index1/3的全部6折风险增加保留。仅支持另登记有界接收IQ的E20身份验证，未实现或启动训练、未证明Ag有效。CPU1.737601s、0模型/网络/checkpoint/更新/目标。[同run报告](../automation_reports/CV-SincNet/20261016-diagnostic-source-ellipticity-crossfit-manysig-m1-r01/report.md)。
