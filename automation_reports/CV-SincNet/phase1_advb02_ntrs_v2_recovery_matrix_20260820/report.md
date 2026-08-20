# ADVB02 NTRS V2恢复矩阵实验报告

## 当前状态

- 状态：`LOCAL_VERIFIED`
- 实现提交：`b8bb34ee299e984dccd52a0a06765d26b3a8419e`
- 目标：按回退诊断报告分离学习率不公平、非严格恒等、独立分类头和旧V1结构的影响，并验证最小共享头V2能否恢复LEO_WEAK性能。
- 数据协议：Phase1 source-only，`L_s/U_s/V_cal/V_select=0.07/0.63/0.15/0.15`，seed=`392034`。
- 训练与最终测试信道：仅`leo_clear_weak`、`leo_low_elev_weak`、`leo_rain_weak`；历史`mixed_orbit`不参与。
- 训练轮数：200；最终checkpoint固定为E200。
- 完成条件：每个训练行均保存E200 checkpoint，并完成clean和三种LEO_WEAK逐场景独立测试。

## 完整矩阵

|行|run ID|profile|唯一方法差异|计划GPU|
|---|---|---|---|---:|
|D0|历史M0 control|复用|同协议Core90基线，不重复训练|N/A|
|M1-DIAG|`phase1_advb02_ntrs_v2_recovery_20260820_m1_diag`|只读诊断|复用历史M1 E200，补raw/robust/fused及门控遥测|4|
|D1|`phase1_advb02_ntrs_v2_recovery_20260820_d1_bypass`|`v2_identity_bypass`|严格旁路，骨干基线学习率|0|
|D2|`phase1_advb02_ntrs_v2_recovery_20260820_d2_bypass_v1lr`|`v2_identity_bypass_v1_lr`|严格旁路，但保留旧V1骨干低学习率|1|
|D3|`phase1_advb02_ntrs_v2_recovery_20260820_d3_v1_fairlr`|`v1_fair_core_lr`|完整旧V1结构与损失，仅恢复骨干基线学习率|2|
|V2-1|`phase1_advb02_ntrs_v2_recovery_20260820_v2_min_shared`|`v2_min_shared_head`|单身份前向、共享CosFace头、无LayerNorm、确定性40维快上下文、最小有界残差|3|

D1/D2/D3/V2-1均完整训练200轮；不会因中途性能低而停止。D0与M1只作冻结历史证据和只读诊断，不重训、不覆盖。

## V2-1冻结设计

- E1–90：残差严格为0，NTRS学习率为0，输出与原始身份路径同坐标。
- E91–130：NTRS残差和学习率线性升至1；骨干始终使用`2e-4`基线学习率。
- E131–200：保持完整残差强度；残差范数不超过身份锚点范数的20%。
- 不使用慢状态、metadata、物理IQ校正、切空间、独立robust head、nuisance因子头和旧安全融合。
- 训练项仅保留主共享头CE、sat-KL=`0.01`、prototype margin=`0.03`和直接残差范数=`0.001`。

## 本地验证

- 39项NTRS模型、训练、评估和launcher聚焦测试通过。
- Python编译和launcher语法检查通过。
- 严格旁路输出与raw logits/embedding位级相等；相同seed下核心参数初始化与关闭NTRS的control完全一致。
- 真实E80 checkpoint无query smoke通过：`missing=12`均为新NTRS参数，`unexpected=0`，旁路输出形状为`(2,6)`。
- 独立审查首次发现1个P0和4个P1；定点复审确认P0及3个P1闭合，剩余D3损失隔离P1随后修复并由聚焦测试证明V1保留原绝对损失、V2单独使用相对损失。原问题相关P0/P1已闭合，不再扩大审查范围。

## 发布与运行位置

- 远端release workspace：`/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_advb02_ntrs_v2_recovery_20260820/b8bb34ee/workspace`
- 远端数据：`/home/szu2070436088/2510044040/CV-SincNet/Dataset_WigSig/ManySig.pkl`
- 训练输出：`/home/szu2070436088/2510044040/CV-SincNet/runs/<run-id>`
- 日志：`/home/szu2070436088/2510044040/CV-SincNet/logs/<run-id>`

## 技术停止规则

只在协议/seed/角色/场景错误、错误release或CWD、输出碰撞、进程归属不清、同类确定性预prediction异常至少重复两次、无法生成E200 checkpoint或独立测试不能闭合时停止对应run。低性能不触发停止，不干预无关任务。
