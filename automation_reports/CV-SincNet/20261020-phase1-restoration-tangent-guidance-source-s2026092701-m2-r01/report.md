# 源 clean 局部敏感度约束恢复

状态：RUNNING / VERIFIED。用户目标为完整目标 low urban≥70%，其余五种 practical 接近原 F clean85.113889%。“接近”暂按≤2个百分点记录，是 Agent 的操作解释。登记时最好 low urban66.384127%，目标仍未完成。

源训练已有 identity/wave 输出梯度累计 RMS 比15.426531；本轮检验固定 clean 参考点的局部敏感度是否能约束身份损伤。这个源证据不证明局部近似有效，也不决定最终性能。当前参数梯度对照尚在目标预测，本轮结构、预算及顺序不依据其目标分数制定。

|固定行|语义约束（先各 F，再三 F 平均）|资源|
|---|---|---|
|既有 per_frame_cap|exact finalcos＋0.3 exact T²KL|仅复用既有完整结果|
|feature_tangent|local feature quadratic＋0.3 exact T²KL|新 GPU6|
|posterior_tangent|exact finalcos＋0.3 local posterior quadratic|新 GPU7|

三套原 F50 CROSS_RX（2026092701/02/03，method4df7b94d0fd8ce29056bd8236364e45afa92e07f）及原默认头全部冻结，不拟合新头。对63000条合法源 L/U clean，逐物理样本计算六坐标对2×256 IQ的 Jacobian。feature 坐标为 normalized160feature 乘固定 Rademacher6×160（符号/√6），再除√2；每 F 用独立 CPU generator seed2026102081＋Fseed，不依赖标签、不拟合参数。posterior 坐标为√2×T×sqrt(softmax(native logits/T).clamp_min(1e-12))，T2，六类原默认输出对称处理。

local=Σ6[Jclean×(restored−clean)]²，三 F 平均，替换且只替换对应一个语义项，另一项及系数0.3保持。feature 是六探针近似，不是完整160维 Jacobian；posterior 是带数值下界的局部 KL 二次近似，不是有限扰动的原 KL。J、clean 及 reference feature 全部停止梯度，无 Hessian、高阶优化或 F 更新。固定缓存只用于源训练；推理仍为原网络单帧 received IQ 的 coarse/corrector，无 J、缓存、F、clean、真实信道/TX/RX/场景、其他帧或状态更新。

每行缓存 J 为63000×3×6×2×256 float32，2322432000 bytes，另有120960000 bytes源feature。493个128样本batch，1479次F前向/189000个F输入，8874次坐标反传/1134000个样本坐标。此额外准备工作与存储如实计入，不能把相同模型训练epoch称为相同总成本。NumPy2/Torch2.1使用经测试的可写frombuffer输入桥；输出缓存沿用已验证tolist桥，无numpy ABI依赖。

两新行严格从原合法 R03（method b01b534f52daf9cae9e940ed6ec6f9243bccbdbe，coarse20/corr20）完整相同state开始，不继承失败精修、control或另一对话权重；2140840参数，无新增NN。jointE10/2470，coarse/corr各累计30轮，60组件epoch，串行继承50pass。仍用原R02 source63000 L6300/U56700、epochs1..10、.15clean/.60residual/.25full、observed+clean、batch256、modelseed2026092701/dataseed2026101951；新优化曝光L63000/U567000，1260000 IQ。

wave仍为 IQ NMSE＋0.1logpower，输出空间逐帧cap复用既有operator：分别求任务sum-loss对估计IQ的gw/gi，s=min(1,sqrt(norm²gw/norm²gi))，i2≤1e-20取1，wave0且identity非零取0，停止scale梯度，(gw+s gi)/实际observed+clean数，再一次网络反传。无冲突投影或放大。AdamW3e-5/warm247/cos3e-6、WD1e-4/clip1/FP32/noTF32保持。记录实际逐IQ scale/conflict/norms、训练loss、成本及源诊断。

复用控制按per-frame cap结构固定为gradient_balance/per_frame_cap（method9866f24dab7097294d585069bfcfe529d3346ade），不是按其目标分数选取，不从它初始化，不重训/重预测。源J的额外成本及损失几何差异明确保留。

source V1800/原F1/七视图只作诊断，无源分数/NMSE gate/排序。两新最终E10都自动完整目标168000/clean＋六practical/三原默认头各自argmax；每行七视图全部IQ/ID/预测固定后，独立无CUDA CPU才连接truth/pairedclean。分别报告逐F/TX/RX/day accuracy、MacroF1、修复/损坏/净纠错、clean差距与同物理full256/prefix232/tail24 NMSE/logpower。不按环境拼接或投票，不以单恢复器seed冒充三个独立训练、新盲测、Phase2或在轨性能。目标分数不回流结构/参数/预算/顺序/选择性重试。

50项聚焦检查PASS；唯一一次独立P0/P1审查PASS，审查者独立复跑50项通过。launcher首步实际R03/三F/四Ls no-query smoke，核实局部J、完整初始IQ/state一致、native posterior IQ梯度、coarse/correction合并梯度、F无梯度、checkpoint往返，0TX/V/target/optimizer，PASS立即训练，不增审批或重复数据验证。

一次N607普通账号预检VERIFIED，原R03合法、新run/release不存在，GPU6/7各24124MB空闲无PID，参数实验GPU4/5实际target PID/CWD/cmd匹配并保留。每卡一个本owner、allowner<2，重叠最多4个本owner GPUworker，source≥18GB/target≥12GB，CPU2线程/scorer最多1。新source评分等parameter_balance终态再与本run共用score.lock；source实际退出才同卡target。非法输入/权限/代码、冲突、非有限/OOM、预算/梯度执行或预测失败时保留产物并失败该行，无自动重试，健康行继续。

一个实际代码配置commit、不可覆盖root、单归档本地/远端SHA比较及remotecompile、真实smoke与实际PID/CWD/cmd/GPU/日志增长随后核实。精确命令、环境/CWD、数据/父来源、路径、seed与逐行矩阵在experiment.json。J/IQ/权重/逐样本预测/原始日志和私有机器状态不进Git。没有向另一用户对话发送指令。


## 实际发布与源约束准备核验

方法commit`2afa1cdbd31e1b2d56214499a797bac00ab64c5c`已推送且远端OID独立匹配；一次归档本地/远端SHA比较、remotecompile/modulehelp PASS。launcher首步实际R03/三F/四Ls no-query smoke PASS：两臂原R03完整初始IQ/state一致，真实源J有限非零、形状8×3×6×2×256，native posterior至IQ梯度及coarse/correction合并梯度有效，F无梯度，新checkpoint往返一致，0TX/V/target/optimizer更新。

GPU6/7两条实际source worker的PID/CWD/完整命令/GPU和日志持续增长已独立核实。此时正在生成63000条源clean局部约束缓存，进度为：

- feature_tangent：实际完成12800/63000条源物理样本的三F坐标Jacobian准备。
- posterior_tangent：实际完成12800/63000条源物理样本的三F坐标Jacobian准备。

缓存进度不冒充完成epoch或性能改善。准备完成后按原固定E10/2470训练、源V诊断及完整目标测试，无额外确认/选模；尚无本轮新目标表现。前序参数梯度run健康GPU4/5任务保持，CPU评分等其终态再共用本run score.lock；复用旧per_frame_cap控制，不重新训练/预测。最终实际梯度统计、准备与训练成本在完整结果节记录。
