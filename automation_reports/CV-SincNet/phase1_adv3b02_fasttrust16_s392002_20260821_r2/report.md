# Phase1 ADV3B02 FastTrust数值修复后16条矩阵预登记

## 当前状态

```text
run_id=phase1_adv3b02_fasttrust16_s392002_20260821_r2
status=RUNNING
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
- 最小修复后该测试及MUSE路由、训练集成、FastTrust协议/速度/launcher聚焦联合测试全部GREEN；聚焦联合回归144项通过。
- Python编译、两个launcher语法检查和16条dry-run均通过；远端dry-run实际生成16个训练命令、16次联合评测命令和64份clean/三LEO分场景输出声明，且没有创建run root。
- N607真实checkpoint CUDA无query验证严格恢复ADV3B02权重，missing/unexpected均为0。初始GradScaler从65536自动校准到8192时出现少量预期跳步；分项未缩放CE与local梯度均有限。连续8个真实L_s epoch的实际更新率依次为95.56%、100%、97.78%、100%、100%、100%、100%、100%，后5轮没有跳步；MUSE头参数产生非零变化，query迭代和target truth读取均为0。

## 远端发布与启动

```text
release=/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_fasttrust_r2_3646fa0b
run_root=/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3b02_fasttrust16_s392002_20260821_r2
dispatch_log=/home/szu2070436088/2510044040/CV-SincNet/launcher_logs/phase1_adv3b02_fasttrust16_s392002_20260821_r2.dispatch.log
```

- release对应Git HEAD为`3646fa0bca943fa5687b396610298369a0f00d90`；自动push后远端`origin/work/cvs-active`OID与本地一致。
- 单一release归档本地路径为`E:\type10-7\release_archives\phase1_fasttrust_r2_3646fa0b.tar.gz`，远端路径为`/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_fasttrust_r2_3646fa0b.tar.gz`；唯一一次本地/远端SHA-256均为`5238e6be12d2a722baf9cde5ac9eafbaed4208385777d4fbfe81822c58c46c94`。
- dispatch PID为`335414`，CWD为`/home/szu2070436088`，cmdline绑定上述release launcher；新run root和dispatch log均为不可覆盖新路径。
- GPU0–7各恰好2个本run的GPU主训练进程，合计16个；16份train.log均已增长，启动错误指纹为0。
- 16/16条均已完成E1。首轮实际optimizer step率范围为97.83%–99.52%；所有MUSE U256候选均为98.55%，U128为99.28%，U384为97.83%。少量首轮跳步来自GradScaler初始scale校准，不是连续零更新；当前各行已到E1–E3，未触发防复发停止规则。
- 启动健康快照：GPU利用率94%–99%，显存4.64–5.83GB/24GB，温度65–87°C；GPU7温度较高但尚无OOM、Xid、thermal failure或训练异常，后续只读监控继续观察。

当前状态为`RUNNING`，尚无E200性能结果。只有每条`final_ssdg.pth`完成clean和三种LEO弱信道测试后，才可进入`ARTIFACTS_COMPLETE`与性能分析。
