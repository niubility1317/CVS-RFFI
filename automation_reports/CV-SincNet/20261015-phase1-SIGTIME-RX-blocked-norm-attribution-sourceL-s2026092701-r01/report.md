# 冷初值 RX 分块与辅助梯度范数归因

状态 PLANNED。上一轮完整默认头三种子证据责任路由未达到总体 84% 目标。本审计依据其源诊断结构而设计：原 INITIAL 资格使用五个 RX 的当前训练包；ROUTED/JOINT 的梯度大小也不同，因而尚不能区分路由与强度。目标测试结果不用于设置本审计的参数或排序。

三条可执行方向：①本轮先做 RX 分块、同标量且匹配辅助 F 梯度范数的一步归因；②独立研究用真实源身份梯度校准 U 的 AdamW 更新方向，须先建立从零训练且完整排除 heldRX 的源折，不能沿用已见该 RX 的终点；③真实跨 RX 判别关系继续作为独立训练方向，先核对既有 PA/identity-difference/集合/观测负结果，不换名重跑。当前只实现①，不额外发布②③。

固定 source-only：5个留出RX×4个原L180cohort×4模式，共80cell。每cell当前训练只用其余4RX的144包，评价仅用原heldbank中的该RX36包；各包均原合法L_s，V/U/target不用。初始化只读新实际LS0基线R02 seed01的scratch0快照，来源/数据角色在加载前检查；不加载训练终点或旧optimizer moments。所有cell从同一冷初值/空AdamW状态独立构造，不串接。

ZERO为原联合CE；JOINT和ROUTED沿用相同同包损失结构、权重.1、容差1e-6、当前4×144固定分母576。JOINT_NORM令loss数值保持原值，仅把其梯度乘以由当前144包计算的stop(norm(ROUTED_F)/norm(JOINT_F))。heldRX的标签/特征不参与因子、梯度或训练状态。W/三个gain的辅助梯度停止，主CE仍联合训练；不拟合任何身份/RX诊断头，actualjoint160保持部署路径。范数控制保持JOINT方向，匹配的是辅助参数梯度范数，不是AdamW实际位移；后者必须单列，不能据此宣称完全消除了优化尺度影响。

每临时步保留原188参数的AdamW lr2e-4/wd1e-4/clip5，主头正常更新。ACTUAL使用afterW和afterfeatures，PRE_W仅替换为beforeW、保留同一afterfeatures和aftergains。真实优化器更新0，160次临时CPUAdamW、900L载入、5942packet网络前向（含2zero-smoke），不保存训练checkpoint；活模型/optimizer/grad/RNG保持。读出只含defaulthead CE、错误、margin和实际F变化，全部80cell/20paired/5RX汇总，不挑优组、不以此自动晋级主训练。

严格边界：这是冷初值一步RX-blocked更新诊断，**不是完成E20的真正留一RX泛化实验**，不能据源CE下降宣称DG。原数据角色与物理ID保持；bank中的评价原始包仅作独立读出，不进入当前训练forward/梯度/校准或模型持久统计。未训练模型不携带训练RX知识，但该有限诊断不替代原要求的整折训练、U/teacher/cache完全排除。

run `20261015-phase1-SIGTIME-RX-blocked-norm-attribution-sourceL-s2026092701-r01`，代码/config由本次Git提交固定，owner `codex/root/RX-blocked-norm-attribution-20261015`。CPU2threads/CUDA_VISIBLE_DEVICES空，普通N607；CWD `/home/szu2070436088/2510044040/CV-SincNet/releases/cvs_SIGTIME_RX_blocked_norm_attribution_sourceL_20261015_r01`，命令 `/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -u experiments/cvs_rx_step_attribution20/runner.py resolved_request.json`。唯一不可覆盖输出 `/home/szu2070436088/2510044040/CV-SincNet/runs/20261015-phase1-SIGTIME-RX-blocked-norm-attribution-sourceL-s2026092701-r01/source_diagnostic`，source依赖为已完成R02不可变release `/home/szu2070436088/2510044040/CV-SincNet/releases/cvs_SIGTIME_evidence_responsibility_train20_20261015_r02`。完整数据/六类seed/物理索引继承字段及单row内80cell矩阵见[experiment.json](experiment.json)和config，均不新增未知seed扫描。

聚焦协议负例和范数/同值检查；唯一独立下一run P0/P1审查；launcher首先严格加载真实cold checkpoint做no-query smoke，PASS直接继续；单archive比较/remotecompile/preflight、PID/CWD/argv/CPU与loggrowth核实。停止：Protocol/role/provenance mismatch, nonfinite or missing native component, wrong norm control/budget, output collision or unlaunchable source execution: preserve outputs, no automatic retry. No performance stop and no healthy-process intervention.。本审计无实际训练，故无最终target预测/评分；整体84%任务仍active，历史正收益和完整负结果保留。

聚焦本地验证 PASS：32 个 source 角色/RX 排除/物理 ID/缺项及范数因子负例；三个辅助条件数值相同、范数控制保持 JOINT 方向并匹配 ROUTED 辅助 F 范数；W/三个 gain 辅助梯度 None。编译 PASS。唯一直接 P0/P1 审查最初发现既有失败根可能被 runner 异常处理覆盖；先复现 FAIL，再将失败处理移到成功新建根后的所属调用中。3 项输出归属回归 PASS，定向复查已解决 P1，保留初始 FAIL 记录。当前 source 梯度和临时读出均使用 eval 模式，不更新 normalization/dropout 随机状态；不等价于真实 20 轮 train 轨迹。真实 CP no-query smoke 仍在 launcher 首步，不以本地合成检查替代。
