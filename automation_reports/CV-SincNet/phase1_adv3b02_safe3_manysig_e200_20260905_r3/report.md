# B_SAFE教师缓存修复与3行恢复发布

## 健康检查与故障

用户要求检查健康、修复失败并重新发布。现场读回r1有效12行继续运行，E19—36；r2中方向约束9行继续运行，E13—16，已越过原E11故障点。B_SAFE三个seed仍在E10结束后退出，错误为`safe region z_id and actual teacher classifier feature spaces differ`。详见`parent_readback.json`。旧r1历史失败12行不重复计数。

本次不停止或热补丁健康21行，不删除旧日志和权重。只在新root恢复B_SAFE三个seed。

## 已复现根因和修复

`CosFaceHead`在eval/no_grad时缓存归一化分类权重，以Parameter的版本计数识别更新。`_update_ema_model`通过`ema_p.data.mul_().add_()`更新参数，绕过版本计数。EMA权重虽然已更新，实际教师logits仍使用旧缓存；B_SAFE一致性检查读取当前权重，因此正确地报出不一致。

修复为现有no_grad上下文内直接`ema_p.mul_().add_(p)`。EMA数值公式、decay、学生优化器和科学参数不变；Parameter版本更新让既有缓存自然失效。不关闭一致性检查、不放宽阈值。

本地先验证失败：`tests/test_pair_ema_cache_recovery.py`两项均失败；单头缓存测试24/24元素不一致，最大绝对误差8.36，真实双骨干教师复现与远端相同ValueError。修复后通过。

旧smoke漏检原因：只执行学生优化步，没有执行EMA更新，因此没有覆盖“缓存已填充→EMA更新→下一次教师分类”的生命周期。新版checkpoint smoke在每个成功优化步后执行真实EMA更新，重新调用教师并验证当前权重和logits一致；六机制、L/U、CUDA AMP及GradScaler均保留。

验证：EMA回归、checkpoint smoke及既有AMP回归共10项通过，包含实际CUDA，无跳过；矩阵与调度器11项通过。运行命令为`conda run -n ssr-gpu python -m pytest`后接上述对应测试文件。保留兼容PyTorch2.1的AMP接口，弃用警告不影响结果。

## 恢复矩阵与比较边界

新run：`phase1_adv3b02_safe3_manysig_e200_20260905_r3`。配置`configs/phase1_adv3b02_safe3_recovery_manifest.json`。

|行|seed|GPU|预算|
|---|---:|---:|---|
|B_SAFE_S392005|392005|1|CORE90初始化，E200|
|B_SAFE_S392006|392006|2|CORE90初始化，E200|
|B_SAFE_S392007|392007|3|CORE90初始化，E200|

与r2原行相比，完整训练argv仅替换run/release/output路径；原始seed、130/70阶段、11/21启用时机、所有损失和原GPU保持一致。从同一CORE90重新执行完整E200；旧final_only未保存E10恢复checkpoint，不假称断点续训。

有效24行的运行状态映射为r1的A_POINT/ASYMMETRIC/POINT_MEMORY/MATCHED_ZERO共12行，r2的TANGENT/ROUTE/TANGENT_ROUTE共9行，以及r3的B_SAFE共3行。合并每GPU仍为3个本轮任务；其他实验保持不动。

**科学比较边界：NO_PROMOTION。**该共享EMA问题也可能影响仍在运行的旧版本教师分类概率。旧21行继续作为原版本诊断结果保留；新B_SAFE与旧版本的差值混合了教师缓存修复和机制差异，不能解释成严格同版本消融，更不能据此晋升默认方法。未擅自重跑全部健康任务；同版本验证需另行固定一致代码矩阵。目标/query只读、source V选择及truth-last边界不变，本次不评分。

## 发布与监控状态

远端项目`/home/szu2070436088/2510044040/CV-SincNet`；代码`releases/<新run>`，输出`runs/<新run>`。现场确认新目录均不存在，Python3.10.19/PyTorch2.1.0+cu121、CUDA、数据和CORE90权重有效。

当前为本地验证完成，等待聚焦审查和远端发布读回。新发布首步执行真实CORE90的EMA＋CUDA AMP smoke，通过后立即发放3行；不会启动第二份健康任务。

Luna每小时检查将跟踪r1有效12行＋r2有效9行＋r3有效3行；历史B_SAFE失败不反复报警。本次只证明修复和执行，正式E11及E200稳定性需后续日志验证。
