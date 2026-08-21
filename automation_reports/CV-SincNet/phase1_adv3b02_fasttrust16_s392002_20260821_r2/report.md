# Phase1 ADV3B02 FastTrust数值修复后16条矩阵预登记

## 当前状态

```text
run_id=phase1_adv3b02_fasttrust16_s392002_20260821_r2
status=LOCAL_VERIFIED
seed=392002
epochs=200
matrix_rows=16
gpu_count=8
rows_per_gpu=2
```

## 修复目的

上一run在系统执行和数据划分正常时出现MUSE本地分类概率在AMP下转回float16、非目标类概率下溢为0、有限forward loss产生NaN backward梯度的问题，最终导致优化器持续跳步。r2只修复该共同数值路径：本地分类概率固定保留float32，FastTrust路由、三头定义、U_s身份规则、损失权重、seed、数据角色、星地增强和矩阵均不改变。

旧run固定为`STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`，其远端run root及全部artifact原位保留。r2使用新的不可覆盖output root，不从旧run恢复任何故障期训练状态。

## 冻结矩阵与协议

- source角色：`L_s/U_s/V_cal/V_select=0.07/0.63/0.15/0.15`，实际预期样本数`5880/52920/12600/12600`。
- 全部行使用seed`392002`和E200；R0从scratch，其余行从`ADV3B02_CORE90_SOFT_E200/best_joint_safe_ssdg.pth`初始化。
- Phase1训练复用CORE90同款LEO_WEAK拼接增强，`lambda_sat_cls=0.68`、`lambda_sat_cons=0`和三段场景日程。
- 每GPU两个实验；GPU3为U128+U384，其余GPU为U256+U256，每卡U batch总和均为512。
- U_s每epoch完整覆盖；U伪身份只能来自FastTrust预定的H/M/candidate路由，U_H星地身份CE仍要求high、temporal stable、三头一致和class cap。
- checkpoint选择固定为`final_only`。训练完成后必须自动测试`final_ssdg.pth`的clean、`leo_clear_weak`、`leo_low_elev_weak`和`leo_rain_weak`，四份逐场景结果不齐全不得标记完成。

## 技术停止规则

- protocol/query泄漏、错误checkout或output碰撞立即停止对应run。
- 同一候选连续两个完整epoch的`train/optimizer_step_applied=0`，视为确定性非有限梯度执行故障；仅停止该run-owned进程树并保留产物。
- Traceback、OOM、`TRAIN_FAILED`、`EVAL_FAILED`或prediction闭合失败按对应技术失败处理。
- 不因中间准确率低、loss走势差或候选性能不佳停止训练。

## 发布前验证

- 新增AMP概率下溢回归测试先RED：三类logit`[0,-20,-40]`产生有限forward loss但输入及五组本地头参数梯度为NaN。
- 最小修复后该测试及MUSE路由、训练集成、FastTrust协议/速度/launcher聚焦联合测试全部GREEN。
- 发布前仍需完成完整聚焦回归、Python编译、launcher语法/16条dry-run及N607真实checkpoint CUDA多批次无query smoke。

## 预期远端路径

```text
release=/home/szu2070436088/2510044040/CV-SincNet/releases/<new-release-id>
run_root=/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3b02_fasttrust16_s392002_20260821_r2
dispatch_log=/home/szu2070436088/2510044040/CV-SincNet/launcher_logs/phase1_adv3b02_fasttrust16_s392002_20260821_r2.dispatch.log
```

真实commit、release归档SHA、CUDA smoke结果、dispatch PID、GPU映射和首次日志增长将在对应证据生成后回填。
