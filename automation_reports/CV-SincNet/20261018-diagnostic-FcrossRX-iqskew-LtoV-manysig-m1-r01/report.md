# I/Q 相对时差：单一源域前提检验

PLANNED／LOCAL_VERIFIED，尚未读取真实新源IQ或获得成绩。固定F/CROSS_RX/defaultcos及用户后续E30预算保持，本阶段0网络训练/前向，不能替代整个Ag+As目标。As三seed适应已确认，clean全seed保持未证实；此新设计仅由已有源数据权限、实际代码范围和独立物理模型提出，不从新目标错误选参数。

已有旧普通Wiener相位噪声不重做。旧SRO对整个复信号统一采样速率，RXMIX是不带记忆的x+b*conj(x)；原DAC inter_skew是复信号相邻采样混合，且p_dac=0。原RX IQ注入/补偿不含此处单独提出的I/Q相对时间响应。差分时差是待验证源假设，不宣称换成真实RX、分离未知TX效应或物理可识别。频率依赖IQ失配的物理可能性参考[IEEE TSP 2021](https://ieeexplore.ieee.org/abstract/document/9314287)，论文不证明当前数据存在此规律，也不证明RFFI身份保持。

唯一假设：同TX/day的RX差异中，I/Q复相干谱的一部分可由一个跨TX/day共享、RX session稳定的相对时差函数预测。对每个L/V cell，以原完整256点的I、Q分别rFFT，固定使用正频率k=1..127（DC/Nyquist数学排除）；先在cell内平均I功率、Q功率和I*conj(Q)，再除sqrt(meanIpower*meanQpower)。不是每包unit-magnitude周期图直接当相干性，不选择频率/窗口。源roleL6300每cell70、V27000每cell300；全部5RX/3day/6TX保留，无训练RX留出/无L_dev。原unitRMS(centerfalse)输入不新增网络归一化。

拟检验函数为 C_r(w)≈C_ref(w)*exp(j*w*delta_r)，refRX1仅是相对坐标，不去掉任何RX。delta固定拟合范围[-.5,.5]采样点（25MHz下±20ns）、101点网格含0；这是预先设定的统计拟合域，不是已经验证的增强强度。每个heldTX/day只用其他5TX×其他2day×全部5RX的L拟合4个delta；V同TX/day的RX1提供输入参照统计，预测其余RX，全部预测再沿RX中心化比较V实际RXcontrast。不同RX包独立，不声称逐包配对。V从不拟合参数。cell内70个L按固定原顺序奇偶各35估计互不重叠相干谱，跨半内积检验真实相干能量，避免把有限样本相干幅值当信号。

唯一四个源风险模型：共享相位律、identity零RXcontrast、任意additiveRXcontrast、TXconditional additiveRXcontrast。后两者也只用固定L训练组拟合，18个heldTX/day统计folds、72个非reference参数拟合，不是18模型seed或18次网络训练。

发布前固定全部条件：identityV_SSE>1e-12；L两半相干内积全局均值≥.0025且每RX>0；shared风险≤.8identity、≤.9additive；至少15/18fold优于additive；每个RX不差于additive；shared≤1.05TXconditional；72个拟合无边界命中。任意失败都关闭此固定函数前提，不增频率/窗口、放宽delay、换refRX/组/seed或阈值救跑。PASS也仅支持下一次独立有限窗/defaulthead身份验证，不能直接称Ag或训练胜出。

这是有限记录的必要统计函数检验：源周期图受边界/包内非平稳/全局I/Q旋转等因素影响，相位律不等于精确有限窗物理分数延迟，也不能唯一分离TX/RX；本阶段不生成任何延迟IQ，不用循环roll/reflect边界冒充长记录。合成周期性shift只验证数学签名，未用于真实观察或训练。后续若PASS仍须用实际有限FIR/有效观察区单独验证身份保持；不反演未知TX硬件。

最小矩阵仅一行IQ_SKEW_PHASE_L_TO_V。原数据源缓存与sourcecontract/VALIDATED_ONCE/capsule/split handles保持，读取原L/V33300包，U标签不访问；cache中其他derivedstate如被反序列化也不使用，不载模型CP或拟合分类头。源方法参数由L决定，V只评价，不读query/targettruth。9项数学/权限测试、本地16916400个合成特征值独立复算已PASS，图形仅在该合成联通测试中mock，真实图尚未产生；AST与真实发布器两段远程脚本检查发布前完成。G0不消费模型，真实CP no-query smoke不适用，不额外加载权重。

Run 20261018-diagnostic-FcrossRX-iqskew-LtoV-manysig-m1-r01；owner codex/root/iqskew-source50；CWD /home/szu2070436088/2510044040/CV-SincNet/releases/cvs_FcrossRX_iqskew_source_20261018_r01；独占输出 /home/szu2070436088/2510044040/CV-SincNet/runs/20261018-diagnostic-FcrossRX-iqskew-LtoV-manysig-m1-r01；CPU2/CUDA隐藏；命令`CVS-RFFI/bin/python -u -m experiments.cvs_iqskew_source50.run`。一次直接P0/P1后写实际代码commit，普通账户preflight、一个archive一次SHA对比及remotecompile，然后launcher首步9项synthetic checks PASS立即继续。停止规则仅所属角色/config/非有限/输出碰撞或无法执行/复算不一致，保留失败不自动重发，不触碰其他健康任务。scientificFAIL正常完成并报告。

预期私有L/V周期图特征、sourceconfig、72参数/18fold/5RX风险、独立复算两张CSV/一个三panelPNG/PDF、完整成本/终态；正式识别成绩和跨seed提升本阶段均不存在。当前代码commit将在发布前由实际Git填入，尚无N607执行。

实现 commit：`6c3cd4c3856df0b8aa64ac6a3eda233ad3070bc4`。9 项单元测试、完整规模合成复算、模拟发布检查及单次独立 P0/P1 审查均 PASS；尚未读取真实源 IQ 或启动远端。

真实源域实验已完成：技术 VERIFIED，科学假设 FAIL（3 PASS/5 FAIL），固定路线关闭；未进行网络训练或目标测试。完整统计、独立复算、成本和图表见 [final_analysis.md](final_analysis.md)。
