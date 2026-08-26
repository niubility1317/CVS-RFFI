# PHASE1_JMRS02_J1_ROLE_SCREEN_S20260824_20260826C预登记与追踪报告

## 一、定点重发

J1-B已因RX1训练梯度非有限而自然停止，prediction未闭合且无性能结果，全部产物保留。C仅修复该系统技术问题：优化器更新前将参数梯度中的NaN/Inf元素置0并逐epoch记录`sanitized_gradient_elements`，随后再执行norm clipping；RX1有效学习率由`3e-4`降为`3e-5`。其他row、source-only数据角色、inner/outer LORO、场景、gate、损失权重、评价指标和晋级门槛不变。

## 二、冻结协议

- rows：`B0,RZ0,RZ1,RX1,D1P,P0`，无联合row。
- 外层7个source receiver LORO；每个outer fold内对其余6个receiver做10 epoch inner-LORO；outer固定40 epoch。
- 审计场景：clean、clear weak、low-elevation weak、rain weak。
- D1P不含线性/对数差分谱比值、相邻频点商或循环roll；P0不产生TX residual。
- V_cal只选择非零rescue-harm gate；held V_select不参与训练或校准。
- 仍是冻结Core90下新增模块source receiver增量LORO，不访问target/query，不是target DG。

## 三、不可覆盖定义

- run ID：`PHASE1_JMRS02_J1_ROLE_SCREEN_S20260824_20260826C`
- 实际修复代码commit：`8f098f46c14a589bfd08fcbb6a4dfbeca7906bff`
- checkpoint：`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3_mechanism32_queue_20260701/ADV3B02_CORE90_SOFT_E200/best_joint_safe_ssdg.pth`
- 数据：`/home/szu2070436088/2510044040/CV-SincNet/Dataset_WigSig/ManySig.pkl`
- GPU：`cuda:0`
- 输出根：`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_jmrs02_j1_20260826/PHASE1_JMRS02_J1_ROLE_SCREEN_S20260824_20260826C`
- smoke根：同级`PHASE1_JMRS02_J1_ROLE_SCREEN_S20260824_20260826C_SMOKE`
- 日志：`/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_jmrs02_j1_20260826/PHASE1_JMRS02_J1_ROLE_SCREEN_S20260824_20260826C.out`

技术停止条件与B相同；若RX1仍在不同inner row重复产生非有限前向，保留现场并停止继续重发，不以跳过RX1或改矩阵掩盖失败。低性能不停止。

## 四、晋级门槛

RZ0/RZ1/RX1/D1P需同时满足三LEO final mean gain>0、clean drop≤0.30pp、四场景receiver floor下降≤0.30pp、至少一个LEO gate coverage>0。P0只比较nuisance proxy MAE与零预测基线。只有RC候选与D1P同时通过才允许后续`BEST_RECEIVER+D1P`新run；C不训练联合。

## 五、本地验证

梯度定点测试先RED后GREEN；J1专项15项通过，`py_compile`通过。B状态为`STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`，不会重复启动。

## 六、发布与启动

- release Git状态：`bd6b56269bdeca9919012fea6981dc9d112cb958`；实际修复代码commit为`8f098f46c14a589bfd08fcbb6a4dfbeca7906bff`。
- 完整Git归档：`PHASE1_JMRS02_J1_ROLE_SCREEN_S20260824_20260826C_FULL_bd6b5626.zip`；SHA-256=`438758bd5e330682e7b65324927fcabcb6589a39bc65233fb539c38b0b56681f`，本地与N607一致。
- 远端release CWD：`/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_jmrs02_j1_20260826/bd6b56269bdeca9919012fea6981dc9d112cb958`；N607原生`py_compile`和`bash -n`通过。
- 启动前C的formal/smoke输出根均不存在，没有既有J1 runner，GPU0空闲。
- C仅启动一次：launcher PID`3574201`，父调度shell PID`3574200`；首次核对CWD/cmdline/run ID/release路径一致。
- 真实checkpoint无query smoke通过；日志进入`fold=rx0,row=RZ0,inner-LORO`。GPU0约848MiB、16%利用率，日志增长正常，无Traceback/OOM/NaN。

当前状态：`RUNNING / REAL_CHECKPOINT_NO_QUERY_SMOKE_PASS / NO_PERFORMANCE_RESULT_YET`。

## 七、运行中完整历史诊断（截至rx1/RZ0完成）

截至当前已完整生成6份训练历史：rx0的RZ0/RZ1/RX1/D1P/P0和rx1/RZ0，均包含6个inner fold与40个outer epoch。RZ0、RZ1、D1P的outer loss分别由1.9486→0.7276、1.8943→0.7307、1.9625→0.7054；P0 nuisance loss由0.01231→0.00585，均无非有限梯度清洗。C已跨过B在rx0/RX1 inner-LORO的原失败点并保存rx0全部5个模块模型，当前进入rx1。

但RX1暴露出确定的数值问题：40/40个outer epoch均记录75816个非有限梯度元素，累计3032640个；loss为1.9345→1.9729，没有形成下降趋势。其6个inner held receiver的rescue/harm均为0，说明canonicalizer实质保持identity；V_cal所谓50% gate coverage仅带来0.00231个百分点表观变化，不能解释为RX1有效。当前梯度清洗避免了进程崩溃，却没有恢复可学习校正。

按预注册规则不因低性能停止C，继续完成其他row和fold；但RX1已提前标记为`NUMERICAL_GRADIENT_FAILURE / NO_SCIENTIFIC_PROMOTION`。

## 八、根因更正（C运行中只读诊断，不修改C）

后续TDD在完全不经过Core90的本地最小图中复现了同一非有限梯度：RX1把零初始化的幅度/相位曲线写成`sqrt(mean(x^2))`，该函数在`x=0`处导数未定义。单步共有1944个estimator参数梯度非有限；C每epoch约39步，`1944×39=75816`，与日志逐epoch计数精确相等。因此“Core90输入梯度奇异”不是现有证据能支持的首要归因，已更正为RX1自身零点范数错误。C是不可变旧release，继续只读运行；不修改、不重启、不覆盖，其RX1结果仍无科学效力。

修复采用`sqrt(mean(x^2)+eps)-sqrt(eps)`，在零点保持数值为0且梯度有限；新run必须先通过真实checkpoint反向smoke，才能判断Core90链路是否还存在独立问题。新run不再清洗NaN/Inf梯度，发现任一非有限元素直接技术失败。

## 九、最终结果与结论

C完成302400条prediction/truth闭合和4个独立评分JSON。B0三LEO均值89.9233%；RZ0、RZ1、RX1、D1P的三LEO增益分别为-0.2275pp、-0.2619pp、0和-0.3307pp，均未晋级。P0 proxy MAE为0.0524383，较零基线0.0532285仅改善0.0007903，约1.48%，只允许nuisance proxy结论。

35份outer history均为40 epoch；RX1七fold合计清洗21228480个非有限梯度元素，结果作废，其他row非有限计数为0。最终不允许`BEST_RECEIVER+D1P`或任何joint。这里的LEO准确率是source RX0–RX6 held-receiver合成弱场景结果，不是RX7–RX11 target-DG，也不等同ADV3B02 target协议。

最终状态：`ANALYZED / NO_JOINT_PROMOTION / NOT_TARGET_DG`。
