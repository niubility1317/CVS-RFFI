# 接收机输入状态只评估恢复（2026-10-31）

新 run `20261031-phase3-receiver-input-s2026092701-r02`，只执行预测与独立评分。原 run r01 / code `ac8bbd27277fcb1f16806c7b8b94e8c3e27e390b` 的 SOURCE_INPUT/TARGET_INPUT 已各完成200步，共400步；两个16参数状态、采样、损失、模型来源及训练产物全部保留，不重新训练或覆盖。

r01 在预测加载器中止，尚未进入候选合成smoke、源预测或目标Q处理，truth读取0。加载器先在CPU重算系数，再与CUDA训练报告逐位比较；同一严格FP32 theta所得FIR最大差为SOURCE 4.656612873077393e−10、TARGET 1.1641532182693481e−10，gamma均相同。实际theta保持一致，问题是派生FP32报告跨设备的精确相等要求。r01保留FAILED_PRESERVED/NO_PERFORMANCE_RESULT，不能记作科学NO_PROMOTION。

唯一数值修复是派生报告按每元素最多2个FP32 ULP比较，keys/shape/floating/finite仍检查；theta本身、arm、原配置、训练预算和checkpoint来源继续严格匹配。真实JSON已在本地复现旧拒绝/新接受，1至2ULP与大错/θ改变的针对性负测已保留。没有修改滤波公式、权重、损失或预算。

r02使用新的Git代码、release/output root和唯一owner。训练配置仍原r01/run/code/输出，预测配置与新产物属于r02；加载器保留原训练身份。入口只有predict→score，不能再启动train。继承训练400步、新theta/NN优化0、新L/U IQ读取0；预测仍需原Q received经过固定DURA，再经两个已冻结输入状态、原F0/PCD。新真实Q F0/PCD各40952、G20476，smoke按实际调用另计。

原[科学计划](PAMF_PHASE3_RECEIVER_INPUT_PLAN_20261031.md)的六方法、三个DURA条件、同物理未来Q、原F0未输入校正的同DURA基准、旧density/12unique tau及全部绝对/相对门槛不变。所有6×3预测后独立CPU三truth各一次；无新U训练、clean学习、未知真TX/K/配额、query拟合、匿名注册或按目标成绩调整。

source V/NV三臂原样重复旧PCD，四未知与03-23缺类宏N/A按原计划；source校准/历史Q曝光、单RX/seed、正式RX24-18未访问保持。r02评估的是r01冻结状态第一次完整测试，不把r01加载故障解释为方法性能。

复用原已通过14unique/41sub；本次仅原故障数值、跨训练/评估身份、eval-only入口及实际checkpoint→预测→CPU评分交接的定向验证，加一次原问题P1定向复核，不重复完整审查或原套件。单N607 preflight、单archive SHA/compile、真实checkpoint无query smoke、PID/CWD/argv/GPU/log增长和独立truth-last按最小流程执行。

若相同技术故障经本次修复仍重现，保留r02并停止盲目重启；其他科学低分完成全矩阵。每卡新增≤2、实际free≥4096MiB，不停止或修改其他owner。完整Phase3目标ACTIVE，性能结论等待实际完整评分。
