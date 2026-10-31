# MILKR 冻结分类头概率核角点验证

R25 检验当前压缩特征核忽略原分类头信息是否影响旧类识别，完整Tri-SIG-TIME SSL、DURA v1.0、原六类神经分类头全部冻结。新几何固定为0.75原signature几何+0.25原分类头概率余弦，不拟合或重训任何神经分类头。不同于早期semantic-residual的固定分数先验，本版概率只作为PSD核几何的一部分，与当前C4moments/CCA64连续特征联合由合法support拟合MILKR。

每记录四次既有F前向的六类logits opt-in保留，T=1逐相位softmax、四相位概率平均再单位化，保持C4不变；旧extract默认仍只返回原phase0logits。源fold模拟old4/new2时，必须先选old_tx四列再softmax/平均，heldnew两列完全排除，不能用六类softmax后mask来获取新类置信度。F骨干本身仍见过源六TX，明确仅新encoder TX/RX-held，不称完整F未知类。target使用实际old6的六列，概率余弦对列置换不变，不依赖真实query角色。

source合法L6300物理记录、既有ep91/92共12600配对，五sourceRX/TX-held CCA拟合均排除heldRX及两TX全部RX/两epoch。TeacherBridge拟合CCA前剔除headpayload，transform后加冻结概率；最终checkpoint仍只有普通OrbitCCA64 bias/projection，无source样本、特征库、类条件统计或head概率上传。原F权重已包含冻结head，没有新NN训练、F/G/head更新0。

K1源域120核/正则规则按四路宏H、更大ridge/固定序选择，ref6 lambda按当前注册类数/6缩放。K>1真实三物理折OOF选120规则，各折tau与系数仅由训练SUP估计，最终当前全SUP重拟合；概率是逐记录冻结特征，无训练态统计。采用全注册SUP全局尺度，不叠加R24局部basis或Fisher网格。源三景/old4new2与目标六景/old6new20差异保留。

源只设合法、数值和真实F/G checkpoint noquery smoke门槛，不设相对收益门槛；固定新机制首次角点证伪，旧负run不改写、不重开、不按目标成绩挑旧源候选。firstRX1-1/P01/MIX01/seed2026092701，K1/20×N2/20，B8+C16=24行。S0/S1严格分开、对应SUPstate共享Q0/Q1；Q1仍仅G输出，G在原IQ上只执行一次再C4，raw不调用G。所有注册类别竞争，query逐条只读。24预测全部固定后独立truth-last，原版同case复用。每corner两S Q0≥原版+5点且Q1>Q0、旧新均值guard≥原版−1点才补剩余12KN；最终all16才其余六RX，不以角点平均代替最终目标。

sourceGPU4、targetGPU0启动前核实容量，不动其他owner。完整commit/命令/环境/CWD、来源/data句柄、路径、停止条件见experiment.json；最小一次P0/P1审查、真实checkpointsmoke、archive传输SHA/compile和PID/GPU/log增长核实保持。技术失败保留，不盲重试。

唯一独立P0/P1审查PASS；head先crop再softmax/排除列零影响/C4均值不变、实际extract opt-in phasepayload、CCA与headpayload隔离/普通参数存档、PSD/query独立/K1缩放、真实物理OOF及既有orbit/source隔离/非法输入共17项相关检查通过。真实F/G/head checkpoint无query smoke为launcher第一步。
