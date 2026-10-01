# CVS整体物理观测身份网络：源训练预登记

状态 ANALYZED，N607正式源训练运行中；clean尚未测试。

仅普通CE、身份网络、无输入/信道增强，scratch、固定L6300/V27000、U56700不使用、split392005；两候选×四modelseed2026092701..04，E200×50/10000更新，batch128、AdamW2e-4、wd1e-4、cosine1e-6、FP32/no clip。无teacher/EMA/继承/域骨干/额外损失。

整个身份路径使用共享固定物理观测前端，后接完整256点时间、源域支持80:160的4×20重复网格和观测变化FFT。所有路径常相位不变；affine额外消除给定输入的仿射相位。普通cosine30、160维、6类，不读取RX/day/truth进行前向或设置分类规则。

|候选|前端通道|参数总量（本地实测）|
|---|---:|---:|
|observable_phase|21|196048|
|observable_affine|13|188240|

物理属性检查、CE梯度、逐包独立、弱信号、数据/继承负测及clean协议已经聚焦验证；Conv2d计数与原Conv1d公式验证通过。冻结后还记录sourceV逐TX/RX/day及嵌入分散，合成AM/AM/AM/PM/RX镜像/多径响应仅做诊断，不是增强或TX参数辨识。实测GPU成本及实际源结果尚为N/A。

固定源选择为四seed最终0.5×源V＋0.5×最差源RX最高优先，性能完全并列后才比成本；绝不读目标评分选结构/候选/超参数。参数轻量次要。

默认收尾由独立子run`20261002-phase1-cvs-observable-clean-manysig-m24-r01`完成，只测试源选中的四个模型，旧20个控制预测只读复用，共24行、同168000物理clean query/6类/7RX，预测完整固定后独立truth-last评分；保留全部seed/RX/TX/CM/Macro-F1及负结果。未选候选不测，不以source分数或理论性质宣称性能目标完成。

声明：变化覆盖整体身份框架，不能归因单个机制；不保证任意RX/多径不变、唯一TX硬件恢复、LEO或新增类/SFT效果。当前已历史暴露clean代理基准，不作新盲测。

[前瞻结构](../../../docs/CVS_OBSERVABLE_IDENTITY_HYPOTHESIS_20261002.md) · [源输入依据](../20261002-diagnostic-rff-preamble-source-manysig-m1-r01/report.md) · [物理边界](../../../docs/CVS_RFF_PHYSICS_AWARE_CRITERIA_20261002.md)。

发布前验证：56项独立相关检查PASS；本地CPU八个一次性模型/24次合成CE更新及冻结变换检查PASS。模型物理与执行权限两个互不重叠的独立P0/P1审查PASS。N607在启动前以其实际Torch运行同一已提交CPU检查，失败则不会启动正式训练。

定点执行复审发现CPU诊断stdout与提交JSON串联的P1；发布前抑制CPU stdout，保留独立JSON产物，生成远端脚本compile通过。未发生远端重复提交。

首次发布在本地git ls-remote（Schannel）30秒超时，尚未创建archive/SCP/远端run；独立N607读回无pipeline/row。发布器改用单命令OpenSSL后继续同一预登记，未启动或重跑训练。

## N607源训练发布已核实

代码`202aed46748856c16b015d94595ecd5364dca394`已push并独立核对远端OID。新release传输校验、远端compile与Torch2.1冷进程八个一次性CPU模型/24次CE更新及物理变换检查通过。独立读回dispatcher PID1130303的CWD/argv；8行各占一块GPU，进程与resolved/log增长已核实，状态SOURCE_TRAINING。实际6300L/56700Uunused/27000V、每轮50步、纯CE/无增强/域骨干关闭与目标访问关闭一致。发布后源码保持不可变，后续clean按另一个不可覆盖run发布。详见[发布后读回](evidence/source_launch_readback.json)。

冻结后数值诊断：未选affine seed02原GPU仿射相位logit误差0.001203179超过原0.001阈值；其他affine行及全部phase常相位检查在阈值内。保留该失败，未改阈值、未重训/重选。补充预登记八个本轮冻结模型的CPU FP32/FP64和GPU TF32设置只读诊断；无正式样本/target访问或模型更新，独占新release与输出。

## 完整源训练与冻结

8 行均完成 E200/10000 步，完整 1600 epoch、80000 step、CSV 和全部 stdout 已解析，无运行期技术异常；实际源数据角色一致，scratch/noaug/nodomain/CE-only 核实。按登记的性能优先规则冻结 `observable_phase`，源选择独立复算完全一致。源性能完全并列后才考虑参数和计算，不使用测试反馈。确认 run `20261002-phase1-cvs-observable-clean-manysig-m24-r01` 已预登记，默认完成 4 个新预测和 20 个冻结控制预测的独立评分。见[evidence/source_selection.json](evidence/source_selection.json)和[evidence/source_completion_validation.json](evidence/source_completion_validation.json)。

## 冻结精度诊断完成

独立读回八行/四精度模式，原模型state未变、无正式样本或target访问。affine seed02 GPU默认TF32误差精确复现0.001203179；禁用TF32后2.62260437×10⁻⁶，CPU FP64为5.32907052×10⁻¹⁵。原0.001阈值失败保留，支持TF32数值路径解释，不能改写成原设置PASS。新diagnostic release提交bc9d522bd；训练202aed467及clean预测设置不变。[原始精度读回](evidence/frozen_numerics_readback.json)。

## 独立 clean 确认完成

选中 `observable_phase` 由独立子 run `20261002-phase1-cvs-observable-clean-manysig-m24-r01` 完成四 seed clean 测试，原 20 预测只读复用，共 24 行、192 条评分。准确率 53.7693% ± 2.5759%，较上轮残差 CVS -24.6850 个百分点（0/4 seed 提升）。本轮没有进一步提高平均准确率，参数更少不能替代性能目标。源 run 未读取 target，未选候选不测试，测试评分不回流调参或重选。见[完整确认报告](../20261002-phase1-cvs-observable-clean-manysig-m24-r01/report.md)。
