# 同曝光密集更新的F-free压缩（本轮R17）

状态：LOCAL_PREPARED。当前目标为原R38 dense_half 794529参数基底的压缩与逐场景0.8pp保持：168000样本×七场景，各场景最多少1344条正确预测，整数500×正确数损失≤4×样本数。原R17恢复器另列历史参考，不替代当前原79万目标；0更改旧run阈值。

源证据来自R16完整30轮训练记录，未读其目标/F分数：全程clip=0，末五轮teacher误差仍降15.16%至16.40%，末三轮斜率为负。不能据此断言容量极限，值得检验固定曝光下的优化密度。R17保持10轮、每行source630000曝光和唯一三stage教师IQ NMSE，将B256/micro64的2470更新改为B64/micro64的9850更新。更新数与实际训练成本不同，绝不冒充相同优化成本。配对clean误差detached仅诊断。

唯一直接父仍为原R38 source E20/4940、method e0868f0a0c3cf056a1753a5b4e5ce4f0bff84381；不继承R16或其他已测学生、不延长它们。复用已核实7/63/30与原R17源祖先；训练0TX标签、0F权重/输出、0V/target，只合法源L_s/U_s IQ和原IQ教师；已暴露benchmark，不称新独立盲测。

|固定行|完整参数|可训练prior|batch|峰值LR|GPU|
|---|---:|---:|---:|---:|---:|
|factor625_b64_lr1|750035|201240|64|0.0001|2|
|factor625_b64_lr3|750035|201240|64|0.0003|3|
|dense750_b64_lr3|689129|140334|64|0.0003|6|

两factor625行同SVD.625结构、同初始化与数据顺序、同曝光/9850更新，只峰值LR1e-4对3e-4；dense750为既有GN4 .75宽、LR3e-4。三行原548795参数coarse/action精确冻结，仅prior训练。SVD left/right权重跨三stage共享且bias独立，dense沿原切片；0IQ校准。训练teacher/student共享一次不可变context，部署仍自带完整context。

固定E10/9850，每epoch985更新，source63000×10、L63000/U567000曝光、teacher/studentprior各630000、共享context630000/studentcontext0。received01..10，MODELseed2026092701/DATAseed2026102001；reset AdamW、warm985/cos1e-5/WD1e-4/clip1、实际尾24归一化。context累计40、prior累计50、component-epoch130、祖先串行pass90、F50E50另计；教师/context/F/head零更新。

每版自动source1800×7诊断/profile，再target168000×clean、三suburban、三urban；全部七视图预测固定后独立truth-last，原F50/defaulthead不重训。三行无论源结果全部正式测试，0源筛选、0目标反馈调参/排序/选择性重跑，负结果保留。逐场景准确率/0.8pp、净纠错、TX/RX/day、clean损伤、IQ/FFT/配对clean几何及完整曲线均报告，长期lowurban70和其他五场景距rawclean5pp单列。

完整模型参数/Conv-Linear MAC（FFT/CG另列）/文件字节/实测完整链延迟/显存报告。16sourceV/batch1/2warmup；单原生float32[B,2,256] IQ输入，完整自身coarse/action/prior与3×2CG，无teacher、clean、标签、场景或状态更新；减参不代表实际加速。

GPU2/3/6接续R16对应行终态且actual owned PID/argv退出后派发，R16+R17共最多3 GPU workers，各卡一本任务；再次查free>=18GB与全owner<2，其他owner保持。R06共享锁max1own scorer/CPU2，新不可覆盖run/release。一次普通预检、归档两端SHA、远端compile；launcher首步真实父/F50/2Ls no-query smoke PASS即继续。技术失败仅保留受影响行，健康行继续，0自动重试/回退/误杀。

REJECTED_EXTRA_GATE：项目八项最小流程覆盖通用设计/计划审批。唯一独立P0/P1限下一真实运行；具体命令、环境、CWD、路径、停止规则与artifact见experiment.json。
