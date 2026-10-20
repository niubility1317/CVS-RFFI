# R16：F-free纯IQ教师压缩与逐场景0.8pp保持

状态：LOCAL_PREPARED。用户最新验收为相对固定原R38 dense_half完整794529参数，每场景性能最多下降0.8个百分点。正式七视图各168000物理样本，最多减少1344条正确预测，整数500×正确数损失≤4×样本数，0均值替代。原R17另列参考，0改变历史实验验收口径。

唯一直接父模型为原R38 source E20/4940、method e0868f0a0c3cf056a1753a5b4e5ce4f0bff84381；继承同7/63/30和已核实原R17源祖先，0继承R13/R14/R15已测学生，非新独立盲测。训练仍0TX标签、0F权重/输出、0V/target优化，只合法L_s/U_s IQ、原配对clean和合法源IQ教师。

源侧假设：精确原父的teacher-stage NMSE为0，但paired-clean目标通常非零，其梯度会推动模型偏离父函数。此前两目标相加既要拟合父又要改写父。R16只优化三个恢复stage对原父IQ输出的平均NMSE；clean IQ只作为原有能量归一化和detached诊断，不作为重建优化项。此选择不依赖F/V分类分数或目标结果。新损失、矩阵、预算、seed、rank与最终轮次在target访问前固定。

|行|完整参数|可训练prior|压缩结构|GPU|
|---|---:|---:|---|---:|
|factor625_fidelity|750035|201240|prior卷积SVD 0.625|2|
|factor500_fidelity|720185|171390|prior卷积SVD 0.5|3|
|dense750_fidelity|689129|140334|GN4 prior宽度.75|6|

原548795参数coarse/action逐位冻结，student与teacher共享一次不可变context，部署仍自带完整context。SVD仅在减少参数时转换，left/right权重跨三个stage共享、bias独立；dense行沿既有GN4切片，不新增IQ校准。所有prior均可训练。

三行固定E10/2470、L63000/U567000曝光、teacher/studentprior各630000、共享context630000/studentcontext0；received01..10、MODELseed2026092701/DATAseed2026102001，reset AdamW1e-4/warm247/cos1e-5/WD1e-4/clip1、B256/micro64真实尾24。context累计40、prior累计50、component-epoch130、祖先串行pass90、F50E50另计。教师/context/F/head零更新。

每版自动source1800×7诊断/profile，固定E10后均target168000×clean、三suburban、三urban；每版全部七视图预测固定后独立truth-last，原F50/defaulthead不重训。报告原R38精确0.8pp、实际升降、净纠错、TX/RX/day、clean损伤、波形主体/尾部/FFT/配对clean特征，lowurban70与其他五场景距rawclean5pp单列。0源分数门槛、0目标反馈调参或选择性重跑，所有负结果保留。

部署只当前原生float32[B,2,256] IQ，完整自身context/prior/3×2CG，无teacher、clean、标签、场景或状态更新。完整参数、Conv/Linear MAC（FFT/CG另列）、文件字节、16sourceV/batch1/2warmup实测完整链延迟/显存一并报告，不凭减参推断加速。

GPU2/3/6各一own/max3，2/3原R15、6原R14对应owned句柄终态且实际退出才交接，free>=18GB、全owner<2，其他owner保持，原R06共享锁max1own scorer/CPU2。新不可覆盖run/release；普通预检、一次归档两端SHA、远端compile。launcher首步真实父/F50/2Ls no-query smoke通过立即训练；prior梯度、精确冻结context、teacher/F/head零更新、严格重载输出一致。技术故障保留该行、健康行继续，0自动重试/回退。

REJECTED_EXTRA_GATE：项目八项最小流程覆盖通用设计/计划审批，一次独立P0/P1限于下一真实运行。具体命令、环境、CWD、路径、停止规则与artifact见experiment.json。
