# PAMF 源模式局部协方差与三类 DURA 测试

状态：PLANNED；当前无新性能。run `20261031-phase3-local-cov-s2026092701-r01`，实际代码commit `4d1d645dd13de875a7f6adf32508a9732ffd3f47`。完整目标ACTIVE，科学指标不降低。

唯一主local_predicted_ratio；原pooled_ratio、原predicted_ratio和local_pooled_ratio为固定控制。只用原源L6300的星地DURA P1经冻结r03导出signal160，拟合每70条90格的收缩完整协方差，alpha70/230。保持原90均值、全局whitener、OE32和原F0/default6不变；trace/D应重构原scalarvariance。模型已知V27000仅在cov冻结后校准两个新六tau，另外两个原tau精确复用；全部tau后源novel3150/cleanQ864/starQ864的既有固定特征重算四列score，三条件全部完成后独立CPU各连接三truth一次。详见 `docs/PAMF_PHASE3_LOCAL_COV_PLAN_20261031.md`。

源RX1/3/4/6/8、2021-03-08/15/23；L六旧设备各1050，V各4500，source novel19-15/19-16/7-1各1050。目标Q为RX24-1(index26)03-15旧六/未知三各96，同864物理记录的clean/star条件均已过同冻结DURA。U384/03-08不读，正式RX24-18不访问。NV/Q历史开发曝光，high未知0/4/2缺19-15保留N/A；不重抽或假装新盲测/更多物理样本。训练/拟合IQ全部固定G10MIX01P1，clean仅测试。

来源为class-density code4af0668、纯源signal r03 codedc01a4、原Tri-SIG-TIME SSL F0 E50/11100、DURA v1.0 R69 E20/4940；sourcecontract392005/VALIDATED_ONCE句柄保持，排除U适配学生和P0开放权重。新NN/optimizer/背景/normalizer更新0，新cov90和tau12标量；真实signal6300、F0/G0，synthetic各2。新统计不等于完全不拟合。部署原834665神经参数外另报local矩阵/原密度/tau字节，source行bank不部署。

主报宏AK/RU/H、CK/同输入原F0closed与损失、AUROC/OSCR、完整TX/RX/day/scene/42组和同ID得失。联合gate：sourceknown12保护，cleanRU95/starRU90、CK90、原F0损失≤2pp与主景覆盖；starAK/RU均不低于原predicted。sourceRU仅诊断，source不通过不取消Q；控制不替主，无U/REG，AN/HOW为N/A。

实际命令：`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -B -u -m experiments.pamf_phase3_open_world.star_local_cov_launch start --runtime /home/szu2070436088/2510044040/CV-SincNet/releases/20261031-phase3-local-cov-s2026092701-r01-release/runtime.json`。

环境/CWD：N607 `/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`，`/home/szu2070436088/2510044040/CV-SincNet/releases/20261031-phase3-local-cov-s2026092701-r01-release`；唯一owner `codex-phase3-local-cov-20261031`。输出 `/home/szu2070436088/2510044040/CV-SincNet/runs/20261031-phase3-local-cov-s2026092701-r01`，日志同root/logs，原模型、输入和历史输出不可覆盖。一个producer GPU自动选择实际free≥4096MiB且本任务新增≤2的卡，scorer独立CUDAemptyCPU；其他owner不停止/修改/重启。实际GPU/PID需启动后读回。

聚焦协议负测、数学与一次producer→score合成集成已完成，结果见本地verification；唯一直接P0/P1审查PASS。launcher首步实际三个CP synthetic noquery smoke，PASS立即继续。发布单preflight/单archive SHA比较/remotecompile，启动独立核PID/CWD/argv/GPU/log增长。非法来源/角色、非有限、输出碰撞或无法产生合法预测时保留、不partial评分/自动重启；UNKNOWN先核原PID/root。科学负结果不停止健康任务。

预期产物：launch/pipeline/configs/logs；predict models/density及local、model freeze/diagnostics、sourceV calibration/4×6tau、known/novel和两个Q predictions、prediction complete/smoke；独立score results/complete；本地observations及完整结果分析。原IQ/权重/运行日志/预测不提交Git。
