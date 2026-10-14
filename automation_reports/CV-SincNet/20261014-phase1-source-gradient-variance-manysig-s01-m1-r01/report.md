# 浅中层真实特征参数的源梯度方差诊断

唯一run 20261014-phase1-source-gradient-variance-manysig-s01-m1-r01，一个source-only row SOURCE_GRADVAR-s2026092701；固定 M12 OLD_GATE_COS50 seed2026092701 final50/11100，仅观察，不新训练、不拟合头、不加载未知父权重。既有source_proof核对实际角色/物理ID/33+17/NoExplicitCFO/终轮与自身来源，旧模型源RX均已见，不能叫留一RX。

源证据依据：已完成source_feature_gradient的norm/cosine诊断不足以回答Fishr所需跨域方差问题。固定 L6300 中每TX6×day3×RX5选2个真实独立包，共180，源选择shuffle2026190202。U None、U标签隐藏，V共享cache物化后丢弃，无V前向/梯度/选模，target IQ/truth/stats读取0。

默认 ResidualPhysicalHead cosine6×160 scale30保持；只对实际 t1.pw.weight3456及t2.pw.weight6912求每包真实CE（smoothing.01）导数，不以batch mean缩小。模型eval，autograd.grad不累积.grad，无optimizer、backward update或辅助classifier。全部参数/buffers前后逐元素核对不变。按源RX5域各36样本，计算无偏坐标方差，raw和减同TX/day跨5RX均值的residual分别报告。仅同一layer内比较；不将均值去除称因果解耦或把方差不一致当作Fishr有效。

代码/config一个Git release HEAD；入口 python -u -m experiments.cvs_source_gradient_variance.probe；环境 ordinary N607 existing CVS-RFFI fullFP32 CPU; CUDA_VISIBLE_DEVICES empty; Torch/OMP/MKL/OpenBLAS2threads；CWD /home/szu2070436088/2510044040/CV-SincNet/releases/cvs_source_gradient_variance_20261014_r01；不可变root /home/szu2070436088/2510044040/CV-SincNet/runs/20261014-phase1-source-gradient-variance-manysig-s01-m1-r01。CPU2threads、不占GPU，不热改/停止/restart其他健康任务。输出raw derivative/真实physicalID留远端gradients.pt，保存后重读并独立重算variance；仅小型聚合与报告Git上传。新epoch0、headfit0，真实源forward180，realcheckpoint no-query smoke2zeroIQ；只源诊断，不追加目标测试。

Stop：Role/provenance/shape/nonfinite/state mutation failure preserves outputs and stops; no automatic retry; no target feedback。预期 preflight.json；perrow resolved_config.json；perrow gradients.pt (remote raw derivative and IDs only, not Git)；perrow results.json；perrow completion.json；completion.json；failure.json only on failure。本地真实native无查询求导、6协议负例与平衡采样/无偏方差公式PASS；一次新增直接P0/P1审查后提交Git，单release归档本地/远端一次SHA比较、remotecompile、真实CP no-query smoke PASS即CPU启动；启动后独立PID/CWD/argv/CPU环境/log增长或终态artifact核实。没有签名/新数据重验/报告hash等额外gate。

唯一直接P0/P1审查 PASS，执行commit读取错误已修复且仅该处针对性确认 PASS；无其余直接问题。用户最新要求以SIG_TIME综合并列统计机制，优先转入该组合设计；本诊断仍保持只读CPU，不改变健康MixStyle训练。
