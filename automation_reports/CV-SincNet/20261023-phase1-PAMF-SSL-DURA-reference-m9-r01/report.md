# PAMF 半监督：单模型与固定 DURA 组合补测

本次按用户新增要求，给全部已预登记的 PSM、RXPSM、VIEWANCHOR 各三个模型种子补上固定 DURA v1.0＋同一 PAMF 的七景结果；单模型结果同时保留。监督 F20、普通 PL F21 的两个 readout 已有同一固定 G、相同测试样本的三种子完整结果，直接复用。具体九行、代码 commit、命令、环境/CWD、资源、数据和输出路径以 experiment.json 为准。

此 run 只做冻结评测，新增训练更新为零，不改变健康的 VIEWANCHOR 续训及其原单模型测试队列。每个 F 必须是其原定 SOURCE_FROZEN E50/11100 学生和原默认头，验证真实三段继承、同 seed/源数据角色，等待原 owner 的完整 raw 七景预测后逐字复用原 npz 中的 raw 决策；DURA 新预测写独立、不可覆盖的 run/output root。不存在按测试表现选择 checkpoint、父模型或种子。

G 始终是用户固定的 R69 stage2 E20/4940、266951 参数 DURA v1.0，来源与 F0 源角色一致；TX 无标签、F-free 的原训练权限保持。执行 Graph-RR：reuse=True、graph=True、native=False，FP32、TF32off；不重新训练、改预算、调精度或回退 native。首个 GPU 步骤用两条合法 L_s IQ 载入真实 F/G 做 no-query smoke，并核对图路径和原 G IQ 误差容限 1e-5。PASS 后立即进入预测。

测试固定 unseen-RX 物理集，raw 与 DURA-LU 各 clean、practical_high、practical_mid、practical_low_suburban、practical_high_urban、practical_mid_urban、practical_low_urban，每景 168000 条。预测配置不含真值，全景固定后由独立 CPU scorer 先核对全部矩阵再接入 truth。三卡 seed 固定 GPU5/6/7，启动时计入既有训练及其他 owner，卡上总计算进程少于两个、空闲显存至少 10000 MiB；每卡最多两个本任务 GPU worker。既有 raw owner 保留单 CPU 评分资源至终态，补测预测可以并行，随后本 run 使用一个 CPU scorer。两次相同技术失败停止新派发，保留健康任务与全部产物。

报告同时给出两张七景均值±SD 表、逐 seed 准确率与 Macro-F1、逐 RX/TX/day/RX×day、对各自监督/普通 PL 对照的配对差，以及 DURA 对同一 F 的纠错、损伤与净纠错。原研发目标仍按 PAMF 单模型判断：clean 相对两个对照均至少 +1 pp，六景均值均严格超过 +1.5 pp；组合结果另列，不能替代单模型验收。该测试基准此前已暴露，不宣称新独立盲测，结果不回流训练、调参或候选排序。

本地 17 项聚焦检查 PASS。一次独立 P0/P1 审查指出新进程尚未注册 CUDA 时的名额预留问题，已修复；唯一一次定点复审确认修复，无新增 P0/P1。卡上已注册 PID 与本控制器启动中 PID 取并集后判定容量，避免超派发和重复计数。真实 F/G source-only smoke 和目标预测均尚未开始。

## 发布与实际运行

VERIFIED：执行 commit `0897d2220f6ce311cdce9991e920ec4071bc66b3`，一次发布包本地/远端 SHA 对照和远端编译均 PASS。独立读回控制器及预测 worker 的 PID/CWD/命令与实际 GPU，确认首条 PSM seed2026092702 的 DURA 预测已在 GPU6 运行；其他行按固定种子卡排队。真实 F/G 的两条合法 L_s IQ no-query smoke PASS，source graph/native IQ 容限1e-5，target/truth读取均为0，固定266951参数G、Graph-RR、FP32/TF32off。每行预测前同样执行 source 检查。健康 VIEWANCHOR 训练保持原流程，未修改其发布。

目标预测、评分与双 readout 汇总尚在进行，不把启动或 smoke 当作性能完成。实际运行证据见 launch_verification.json；私有路径、账号、原始运行日志和训练权重不进入 Git。
