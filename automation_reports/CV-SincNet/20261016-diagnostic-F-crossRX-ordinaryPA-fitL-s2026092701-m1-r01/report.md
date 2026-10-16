# F＋跨 RX 同 TX：普通 PA 增强的同抽样身份反事实

状态：PLANNED，未启动。此项审计现有普通增强，不引入新的 PA 损伤，不是新 A_g 效果实验。固定谱滤波候选已因有限 IQ 的 PSD 保真失败关闭，不调参救跑。

唯一假设：已启用的 ordinary PA-normal 在真实触发包上造成可避免的身份错误。普通管线实际为 DC/taper/循环移位/gain/phase/CFO/Wiener/multipath/AWGN/DAC/PA；PA 位于末尾、DAC p=0、class_signature=false。概率按实际日程为0.014至0.049；Rapp的sat0.6至1.1及p2至5未随强度缩减。该继承配置没有获得身份保持证明。

## 固定最小源矩阵

只用自有冻结 F＋CROSS_RX B00/seed2026092701/fixedfinal20，既有三路SIG_TIME、统一Sinc、joint160、NoExplicitCFO及必要补偿、原默认cosine头保持。checkpoint来自20261016-phase1-F-crossRX-satmix25-sourceRX1-e20-s2026092701-m2-r01的B00，scratchE20/4440更新，无祖先、目标接触或目标选模；实际L5040/U45360/V21600契约与当前源折一致，不加载E50/E200或其他来源。它仅作冻结诊断，不初始化新训练。原父成本E20/4440/4686678NN另列。

复用既有source_contract与SOURCE_GEOMETRY_AUDIT_COMPLETE缓存、EXACT_MATCH角色结论。在任何IQ统计、变换或推理前排除heldRX1，fit ID严格等于冻结模型L5040，RX3/4/6/8×day1/2/3×全部6TX×70。未用U、V、heldRX1或target；缓存可反序列化其他源角色和旧特征但不计算使用。该fit分析不能代替未见RX验证，不能冒充完整L6300或L5580/Ldev720。

raw只推理一次。对全部20轮原生增强日程，使用实际checkpoint args调用native build_aug_base_cfg/make_augmentor/configure_augmentor_for_epoch。每轮Python/Torch seed=20261016241+epoch，固定fit缓存顺序及128/末48批次。这是新诊断抽样，不伪称重放旧训练随机轨迹。每批只调用一次原生augmentor，从post-general-channel输出捕获完整prefix，再取得PA后的full；prefix即只旁路PA的反事实。未重抽通用噪声/多径、未强制PA触发、未更改操作范围或追加RMS归一化。记录真实strength、selected输入、subchain调用、变动包、全包及80:160参考窗MSE、功率比。未触发包full/prefix须逐位相同，NN eval不消费后续增强RNG。

raw5040＋20×5040×2=206640源默认头预测，加2合成CP smoke=206642实际NN包前向；100800普通增强包，实际PA触发与变动数读回。0训练/头拟合/teacher/As/额外分类器。局部验证1344合成增强包，0真实IQ/CP/NN，11负例PASS；全部20日程canonical native输出逐位一致、Python/Torch RNG终态相同、同起始随机状态noPA输入等于捕获prefix。

## 预登记判断

在所有20轮汇总的真实PA变动包上，至少1000个观测，旁路PA相对full的准确率恢复≥2个百分点；至少3个RX恢复为正，任一RX不低于−1个百分点，每个day不低于0。记录repair与new break、各TX负结果、原始及full/bypass的accuracy/MacroF1/CM/CE；全部20轮均报告，不挑强度、seed、RX或子链。每物理包重复20次抽样相关，不用朴素包级p值宣称独立证据。结果不满足即关闭这项PA伤害训练依据，不改阈值或救跑；失败也不能证明PA身份保持。

通过仅支持另行登记同预算scratchE20普通配方消融，仍须留出RX的真实clean正收益和practical保护，不能把源拟合恢复直接称A_g泛化有效。当前不自动训练或启动target。主联合目标lowurban>60%及suburban接近clean仍未达成；原始B00/B10/B01/B11要求保留。

## 发布与产物

唯一owner codex/root/F-CROSS_RX-ordinaryPA-20261016；普通N607账号，CPU2线程、CUDA_VISIBLE_DEVICES空，不占GPU槽或干预健康任务。CWD为PROJECT/releases/cvs_F_crossRX_ordinaryPA_fitL_20261016_r01，命令CVS-RFFI/bin/python -u experiments/cvs_pa_counterfactual16/runner.py。精确路径、数据与六类seed及逐行矩阵见experiment.json。代码身份以实际Git commit/release_commit/resolved为准。一次N607 preflight、一个归档单次SHA对比、一次远端compile；launcher第一步真实CP合成无query smoke，PASS立即继续；独立PID/CWD/argv/CPU环境/log增长核验。

角色/CP/非有限输入、实际抽样或成本/状态不匹配、输出冲突为技术失败，保留产物无自动重试；科学负结果不停止或热改，全部20日程完成。保存resolved/smoke/fitID/raw与20轮全部logits预测及真实触发/变化记录、epoch充分统计、result/completion或failure、完整stdout。预测完成后独立重算全部身份指标、触发计数、RX/TX/day配对和固定判断。IQ/CP/逐包预测及日志不入Git，仅汇总报告与代码/配置/登记入Git。
