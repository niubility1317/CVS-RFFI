# 任务强度缩放参照的整网源剂量

PLANNED，仅在合法源L/U上校准，不创建optimizer或新训练模型，不读取V/目标或U真值。机制依据为已完成的 [冻结梯度诊断](../20261026-diagnostic-PAMF-RXReference-alignment-m1-r01/report.md)，旧失败参照矩阵不扩种子，原性能目标保持。

固定同契约原ordinary-PL own E20+10 E30，student seed2026092701。L参照只复用可见L6300构建的own E30特征，U软目标/接受集合与三个源G视图复用原raw三E30教师缓存。加载检查原E20/own10、源物理角色、默认头、native U/EMA关闭、参照/cache同contract/同G；VALIDATED_ONCE不重验、不加hash链。

12批L128/U256、loader seed2026092701。原L保持完整六前向768包、LS.01、ordinary/raw RDrop、As真实CE/KL0、浅中层CRIC.1；原U主项为自身eval argmax/confidence≥.95、raw CE/weight1、完整batch平均。L与raw U主梯度分开计算后相加，no_native_U仅避免L钩子重复计算该U主项。

| 新增分支 | 两次输入 | 目标 |
| --- | --- | --- |
| GUniform | 相同G(scene_k)两次 | 原跨RX参照软CE |
| GTaskScaled | 相同G(scene_k)两次 | 按默认头硬身份任务范数缩放的参照软CE |
| RawTaskScaled | 相同raw两次 | 同一任务缩放公式的输入控制 |

三者共用raw三教师同类且各≥.95的q/mask，新增mask另外排除首个排序源RX1，原主项仍使用全部源RX。k=(step−1)%3，复用low suburban/mid urban/low urban缓存；无G新前向或恢复后噪声。每分支先平均两次loss，最后按完整U batch平均。

缩放系数为 `stopgrad(min(32,||g_id||/(||g_ref||+1e-12)))`。g_ref是参照CE对同一次forward特征z的梯度，g_id是现有默认头对raw伪身份argmax(q)的CE对z的梯度。头在同一z上通过实际encoder.classify_features重算，并核对与原tx_logits一致，避免图中等值别名不相连。权重detach，不把额外梯度传到默认头参数；头仍由原主项联合。报告接受数量、实际权重和/均值/最大值与触顶率，不能把接受数当成真实U精度或等量利用。

每变体前恢复相同post-primary RNG与临时buffers；批后恢复全部named_buffers，包括非持久bank。student/三冻结教师的参数和注册buffers最终不变，参数.grad仍None。共60次整网autograd.grad，另96次特征autograd；student包前向33792，L2真实权重smoke另列，G前向0。

共同剂量在本run结果前固定为 `min(.25,.10*median(primary_norm)/max_variant median(extra_norm))`，三个分支使用同一值。范数对象为全部实际可训练参数的preclip物理梯度：Sinc两个CENTER_25K坐标梯度除25000，其余原单位，匹配CENTER_UNITS全局clip5；不是AdamW更新保证。12批所有norm/cosine须有限、norm非零，不用中位数掩盖单批坏值。无临时目标分数、候选重排或旧0.071458剂量回退。

launcher先真实L2检验own E30、默认头/特征兼容和任务缩放可微分图，通过即继续。两项新剂量负向检查与模块compile PASS；基础旧/新关系数学九项此前PASS。本候选一次独立P0/P1审查结果发布前追加。

命令：`PYTHON -u -m experiments.pamf_channel_pl.task_scaled_source dispatch --config PROJECT/releases/20261026-diagnostic-PAMF-TaskScaledRef-dose-release-r01/runtime.json`。N607既有普通账户Python/Torch2.1，FP32/TF32off，CWD为该不可覆盖release，输出 `PROJECT/runs/20261026-diagnostic-PAMF-TaskScaledRef-dose-m1-r01`。GPU0一条worker，准入时全卡CUDA/pending<2、空闲≥10000MiB，CPU2threads，保护其他owner。Git commit为代码/config身份，一次archive SHA传输比较、远端compile和启动后独立进程/CWD/argv/GPU/日志增长检查。

首个来源/角色/head图/smoke/非有限/持久状态或技术错误停止本run，保留部分产物，不自动重提，不修改其他健康任务。预期smoke.json、progress.json、gradient_complete.json、complete.json、pipeline_state.json与公开12批统计。本run不能说明新训练或目标收益；实际系数与下一单种子匹配矩阵另行冻结后执行。

发布前一次独立P0/P1审查PASS，无P0/P1；重点核对真实默认头重算、图保留/停止梯度、原L/rawPL主项、完整buffers复原和整网求导计数。两项新剂量/负向检查、基础关系/任务缩放九项检查与模块compile PASS。真实L2 checkpoint/head/graph smoke仍为launcher首步。
