# 普通伪标签与部署融合特征跨 RX 对比

LOCAL_VERIFIED_NOT_LAUNCHED。48 项相关检查及一次独立 P0/P1 PASS；真实原 E20 no-query smoke 为首个空闲 GPU 步骤，验证实际融合特征及签名分支梯度后立即重载父模型继续，无额外审批。

固定 HPL_E20_NAMS_JOINT6G、seed2026092701/02/03，原同 seed F21 E20 独立 native10＋20 到 E50／11100，13L＋37LU。仅原 E20 初始化，严格检查实际 payload、scratch 继承、原物理数据角色和默认联合分类头；不继承原 E30、前一候选或目标挑选权重。沿用 F0 VALIDATED_ONCE L6300／U56700／V27000，U TX 隐藏，V 不更新参数、归一化、EMA、原型或头。N607 普通账户、CVS-RFFI、FP32/TF32off、每 worker CPU2；命令、CWD、GPU、不可覆盖路径和预期产物见 experiment.json，实际代码配置以发布 commit 固定，单 release archive 一次本地/远端 SHA 对照、远端 compile，独立进程/CWD/命令/GPU/日志增长回读。

源依据：原 E20 完整 V27000/seed、0.95 伪标签精度99.9002%／99.8687%／99.9164%，实际 U 精度未知。固定原 E20 的 U960＋V720/seed 无更新配对诊断显示现有统计混合在 U 浅中层平均位移约6%至9%，接受样本无预测变化，伪标签 CE 仅约0.003至0.004；最终160维融合特征仍有条件于类的源 RX 描述性关联。style后继续使用原 IQ RX 标签，跨 RX 混合本身就会改变这种关联，不能将其降低解释为模型已学会不变性或因果去接收机证据。U 按硬伪标签分组，V 的真标签仅用于这次不更新的诊断。源诊断详见 source_style_geometry_E20.json 和原预定计划；历史目标暴露如实披露，新 E20NAMS／Guard 目标分数未读取。

唯一新因素：保留 GuardNAMS 的0.95门控、权重6、LS0.01、ordinary/raw RDrop1、As KL0、CE剂量1.25、浅中层跨 RX0.1、无 U 信道、六 U 前向共享 accepted-only 同伪类跨源 RX 统计混合（p0.5、Beta0.1、eps1e-6）和完整 L 梯度保护；在同一前两个 ordinary U 学生前向上捕获实际 CompositeReference.features 的 joint160，增加原生同伪标签跨 RX 对比。单 joint 系数0.05（等于原0.1两层平均的每层系数），除原 CE 归一化1.25，随后整体 U 乘6。原生温度0.2、分离的 batch中心/尺度、accepted-only 同伪类不同源RX为正、其他类为负；保留浅中层，不强制原始接收机响应参数相同。

实际融合 features 在浅中层后加入可学习 SIG_TIME_X3 分支。临时实例 MethodType wrapper 只覆盖这两次已有 U forward，保存并精确恢复原实例属性；不使用会被 .classify 直接调用绕过的分类头 forward hook。无额外前向、teacher/L捕获、新分类头、推理参数或G训练。真实原E20 smoke必须显示实际 [256,160]×2、512条复用packet、有效跨RX anchors、签名 sequence_pool 参数的辅助梯度非零有限、原L768/U1536/As raw128/Uphysical0与LPractical64、共享style plan及临时 wrapper 退出、保护后完整参数梯度与显式期望相同。探针状态丢弃，正式行重载原父权重。

每模型新增30轮/6660步，U1701000，joint6660次/捕获13320批/3402000条已有Upacket；这些属于原有10206000个新学生U视图，不增加数据或前向。累计 U/teacher2097900、studentU10602900、rawAs850500、style39960、共享plan6660。own10/20两段实际状态分别累加，原生 CENTER_UNITS AdamW、cosine、全模型物理 clip5/decay保持，完整L anchor/t2/cls_head/other投影与U/L范数帽1保持。额外joint对比反向计算和保留图有成本，报告实测父/增量/累计墙钟，不称同FLOP。无 Adam、L损失或目标性能单调保证。

GPU5/6/7一seed一卡；每seed只接替 GuardLS 全部GPU预测完成的卡位，且 GuardNAMS已派发。旧两臂尚未CUDA的PID与新child预留计数，每卡最多2本任务、含外部最多3进程、派发前空闲≥10000MiB。唯一CPU scorer排 Guard6 终态后；不停止、重启或热改健康任务。两个相同技术失败停止新增派发、保留健康进程及产物，不按目标结果停止、调参、排序或选择性重跑。

训练固定最终E50学生，自动raw和同一固定DURA v1.0各clean＋六practical，每view168000，三seed共42cell；该row所有预测固定后独立truth-last。报告每seed/场景/TX/RX/day/RX×day、Macro-F1、修复损伤/U曝光/实际成本；分别比较固定F20/F21和同恢复读出。原raw验收同时clean≥+1pp、六景均值严格>+1.5pp，DURA只作独立参考，G权限和Fseed SD范围保持，不套旧F50时延。已有历史及E20LS2目标结果已观察，不称独立盲测；新假设按固定源证据在新E20NAMS/Guard目标分数读取前冻结，goal ACTIVE。
