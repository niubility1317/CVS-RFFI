# 既有三个冻结模型目标评估预登记

run_id：`20261019-phase1-existing-original-eval-m3-r01`，唯一owner=root-existing-eval-20261019。父实验为 `20260930-phase1-three-original-manysig-m3-r01`，原源训练代码commit `1eb16044350101c8d5eb0a4fda2db8447cf3b8a4`，仅隔离NumPy兼容准备脚本后续提交为 `e4ed37ac80696d7d49ed7ebe64e0d0feabb913b4`。原launch plan与完整resolved config记录仅source训练；ASKNet/WaveMLP从零训练，DIFL学生仅继承同run、同源划分的教师。未加载作者权重，未访问目标选模。

本次只补全原固定checkpoint的测试，不重训、换头、选候选或按目标结果重跑。原模型代码直接引用父release相同CWD，原checkpoint保持只读。各行真实路径、模型配置和checkpoint在experiment.json引用的私有runtime.json中，预测前再次核对其源数据契约和继承关系。

| row | 源训练 | 冻结选择 | 本次目标 |
|---|---|---|---|
| asknet-s2023 | 9RX、day1/2、raw前100包；70/30、seed2023；实际41轮 | 源验证NLL最小 | 原round0的3个未见RX、day3/4、3600包 |
| wavemlp-s2023 | 同上；实际39轮 | 源验证NLL最小 | 同上 |
| difl-s300 | 9RX、equalized day1、seed300随机200包；80/20、split seed32；教师/学生各100轮 | 各阶段源验证准确率最高 | 相同3个未见RX、equalized day1、相同200包、3600包 |

目标RX固定为1-1、1-19、2-1，源为作者顺序其余9RX，TX为原6类顺序。ASKNet/WaveMLP复用IB-RFF round0固定匿名IQ/ID导出；DIFL独立导出对应均衡day1目标，不能混用。evaluation seed20261019只用于匿名顺序；model/split/data等沿原行，新增优化/训练预算为0。无support/K或LEO，此任务是原实现全监督clean补测，不是CVS弱标签或卫星部署实验。

普通N607账户、GPU2单进程依次执行3行，每GPU总进程≤3，不影响健康SDG训练。沿用父环境CVS-RFFI Python/torch2.1+cu121及父release隔离NumPy1.26.4。新run/release/log根独占不可覆盖。命令模板为 `CVS-RFFI/python -u RELEASE/evaluate.py launch --runtime RELEASE/runtime.json`，CWD为本次release；精确路径、GPU、代码commit/argv见私有runtime引用。

每行自动执行真实source checkpoint无query烟测→对应匿名query准备→原头预测→全部3600条prediction固定→独立truth-last评分。所有3行预先固定，逐行评分不修改后续行。报告accuracy、Macro-F1、混淆矩阵及逐RX/day；只作同条件证据比较。单折不能验收论文完整均值；DIFL特征取点仍有明确缺口，不依据高分消除该限制。

技术停止仅限本run来源/契约不匹配、错误RX/day/标签、输出冲突、不可执行、非有限logits或无法闭合法预测/错误truth连接。低分不停止。保留失败/partial及旧父产物，不修改或删除旧权重/数据，不停止其他任务。

发布流程限最小白名单：新评估入口聚焦检查、一次仅针对新预测/评分入口的P0/P1审查（不复审原方法）、代码commit、一次N607 preflight、一次archive传输SHA对比、远端编译；启动后核对PID/CWD/argv/GPU/日志增长。预期每行smoke.json、query_prepared.json、predictions.npz、predictions_complete.json、scores.json、completion.json，整批completion；全部真实生成并独立复算后才能记为完成。
