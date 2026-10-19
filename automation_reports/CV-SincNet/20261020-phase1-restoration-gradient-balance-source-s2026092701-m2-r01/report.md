# 整批与逐帧的识别梯度范数上限

状态：RUNNING / VERIFIED。目标仍是完整目标测试low urban≥70%，其余五practical接近原F clean85.113889%（Agent暂用≤2个百分点口径）；分别报告同物理clean的波形差距和冻结原默认头真实TX识别/净纠错。当前已验证最好low urban65.309127%，目标未达成。

source依据：当前gradient_projection的scalar_sum控制行E5/1235、630000条训练IQ，identity梯度norm²总和1191929.8158283，wave为5008.57419455，累计RMS范数比15.426531。该统计来自合法L/U训练，不是逐帧或逐step比，不证明幅度失衡就是性能原因，也不是目标证据。source_gradient_evidence.json仅含聚合统计，0新模型前向/更新/目标读取。新范数约束在该控制行新目标准确率被访问前固定，不热改其或投影行。

|科学比较行|执行|来源|
|---|---|---|
|scalar_sum|复用已登记健康行，不重复训练|20261020-phase1-restoration-gradient-projection-source-s2026092701-m2-r01/scalar_sum，method9f2d8f65e507809de396af5fdb89f11b749759b1，固定最终E10/2470|
|global_cap|本run新训练，GPU4|每个observed+clean整批的范数约束|
|per_frame_cap|本run新训练，GPU5|每个恢复IQ帧的范数约束|

两个新行都从原合法R03（coarse20/corr20，method b01b534f52daf9cae9e940ed6ec6f9243bccbdbe）完整相同state初始化，2140840参数，无新模块；不从当前控制或任何失败精修权重初始化。原wave=paired IQ NMSE＋0.1logpower，identity=三原F各自final160cosine＋0.3×T²KL（T2）的均值，损失数值不变。两者先求当前恢复IQ2×256实坐标上的任务sum-loss梯度gw/gi，s=min(1,sqrt(norm²(gw)/norm²(gi)))，然后(gw+s×gi)/实际IQ批量向原恢复器反传一次。global用整批norm²总和、同一个s；per-frame各帧独立计算s。identity norm²≤1e-20则s=1；非零identity而wave为0则s=0。s停止梯度且只缩小不放大identity方向，不作冲突投影。第一任务求导保留共享图，无Hessian/教师更新。

本方法是source输出梯度的上限，不是参数裁剪、论文GradNorm或PCGrad。global约束批量总体，不保证各帧identity≤wave；per-frame可能更大幅削弱总体identity。共享Jacobian、AdamW及clip也不保证参数步或最终波形/识别改善。记录calls/IQ/conflict/scaled、observed/clean缩放数与scale和、scale均值/最小/最大、前后dot及wave/identity/identity-after norm²。零实际缩放是科学结果，不作资格或技术失败门槛。

scalar控制已在本轮设计中明确固定为前run的scalar_sum E10/2470，与本轮相同原R03、数据/缓存、seed/顺序/路由、paired loss/三F、AdamW日程和分离输出梯度计算。仅在它完成后复用已有完整源与目标结果进行匹配比较，不重新预测或按分数选控制，不把它的权重传给新模型。若控制技术失败，保留并报告缺项，不能静默声称完整匹配比较。target_matrix明确新目标预测仅两新行，external_control为只读比较引用。

原63000 L/U与R02缓存epochs1..10、.15clean/.60residual/.25full、observed+clean等权、batch256、modelseed2026092701/dataseed2026101951保持。新两行各jointE10/2470、L63000/U567000优化曝光、1260000IQ；原coarse/corr各累计30轮、60组件epoch、串行继承50pass，无新增网络epoch。重置AdamW3e-5、warm247/cos至3e-6、WD1e-4/clip1、FP32/TF32关闭。原F50 CROSS_RX三seed2026092701/02/03及原默认头（method4df7b94d0fd8ce29056bd8236364e45afa92e07f）冻结参数/统计，0L/U TX真值/F更新/新头拟合。复用同数据契约与来源，不重复数据验证或增加哈希/签名链。

推断仍为原网络当前单帧receivedIQ一次coarse/corrector，无运行F或梯度、normcap、信道oracle、RX/场景、clean、其他帧/视图、迭代或状态更新。source V原1800条/原F1/七视图只诊断，无早停/资格/NMSE gate/排序/状态更新。两新固定最终E10都自动完整168000物理目标clean+六practical/三原默认头，各自argmax，不投票或按环境拼接；每行七视图IQ/ID/预测全部固定后独立无CUDA CPU才连接truth/pairedclean。报告逐F/TX/RX/day accuracy/MacroF1、修复/损坏/净修复、距clean差距、无oracle对齐full256/prefix232/tail24 NMSE/logFFTpower均值/中位数/P90及耗时/显存、相对复用标量控制的差值。目标成绩不回流配置/预算/权重/顺序/选择性重跑，一个恢复器seed与三读出不冒充三个恢复器seed、新盲测、Phase2或在轨性能。

50项聚焦检查与唯一一次独立P0/P1审查PASS（审查者独立复跑50项）。launcher首步实际R03/三F/四source Ls no-query smoke：初始完整IQ/state相同、有限非零merged coarse/corr及原头到IQ梯度、F无梯度、新checkpoint往返，0V/TX/target/optimizer，PASS立即训练。一次N607普通账号预检VERIFIED，原R03合法、新root不存在、GPU4/5各24124MB空闲无PID、空间≥8GB；GPU6/7现有source PID/CWD/cmd LIVE并保留。每卡一条本owner、计入allowner<2，重叠最多4本owner GPUworkers，source≥18GB/target≥12GB。CPU2线程/scorer最多1，新source scorer先等前gradient_projection complete/failure，再与本run target共用score.lock；实际source进程退出才同卡target，不等另一新行。

停止规则：非法来源/权限/代码、冲突、非有限/OOM、预算/缩放执行或预测失败时保留产物并技术失败该行，无自动重试，健康行继续，不停止/重启/热改他人任务。一个代码配置commit、不可覆盖root、一次归档本地/远端SHA比较及remotecompile/import，随后核实PID/CWD/cmd/GPU/日志增长。命令、环境/CWD、矩阵、路径、资源、停止规则和产物见experiment.json；原IQ、权重、逐样本预测、日志及私有运行状态不进Git。另一用户对话未被发送指令或改动。


## 实际发布与首轮核验

方法commit `9866f24dab7097294d585069bfcfe529d3346ade`已推送并独立核实远端OID，一次归档本地/远端SHA比较及远端compile/import均PASS。实际R03/三F/四Ls no-query smoke PASS：两臂完整初始IQ/state等于原R03，coarse/correction合并梯度及原头到IQ梯度有效，F无梯度，新checkpoint往返一致，0V/TX/target/optimizer更新。两臂实际2140840参数，无新增模块；PASS立即训练。

两条实际source worker的PID/CWD/完整命令/GPU4/5及持续日志增长已核实，三F参考缓存各63000条。实际范数约束/缩放计数：

- global_cap：1/10轮、247/2470步，梯度合并调用247、IQ126000条，实际缩放126000（observed63000/clean63000），平均scale0.046522301、范围0.029634306至0.079243258，wave/identity/缩放后identity norm²总和996.19665/503684.15/996.19664。
- per_frame_cap：1/10轮、247/2470步，梯度合并调用247、IQ126000条，实际缩放50262（observed41542/clean8720），平均scale0.74880064、范围0.00275951至1，wave/identity/缩放后identity norm²总和995.95361/683227.22/521.45933。

首轮核验时尚无新源V或目标准确率。实际缩放不代表性能改善或参数步波形保证；两臂固定E10后分别自动完整目标测试，source V不作资格筛选，用户目标仍未达成。最终状态与结果以完整结果节为准。前一梯度投影的GPU6/7目标进程保留，CPU评分等待其完成后执行；复用其scalar_sum固定控制，不重复训练/预测。
