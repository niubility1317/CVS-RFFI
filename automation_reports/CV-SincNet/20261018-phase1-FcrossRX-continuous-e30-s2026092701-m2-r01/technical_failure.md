# P52 技术失败与边界修复

状态：As 行技术失败，匹配性能比较未完成；健康 NoAs 行按原计划收尾。不是科学负结果，不据此否定连续 E30 假设。原 run/release 及 checkpoint 不改，不自动重跑。

实际训练代码仍为 04ac1098c8c23ec7dab655a1a1317e7363ad580c。B01 完成 E20/30、4440 次更新、3409920 个已记录 L 前向、284160 个 fullAs 训练包，在下一步 E21/4441 调用旧 fullsource_concat20.channel.Practical.apply 时被 E20/4440 guard 拒绝。日志明确 ValueError: Finite real L training IQ/declared E20 only。没有 final_ssdg.pth；initial_smoke.pt 仅是初始 smoke 权重，不能视为 E20 checkpoint。失败步的部分前向及所有验证/U成本仍待完整计数，不把已知 L 数当成全部成本。

根因在训练 adapter 扩为单段 E30 后仍继承旧信道包装器的 E20 执行边界。原短 smoke 没覆盖 E21 这条路径，先前审查也未发现。实际信道引擎和 received IQ 并未改变；失败不涉及真值泄露、非法数据或正式评分。

本地后续修复为 experiments/cvs_continuous52/channel.py：原 full practical 的物理、随机流、session与元数据逻辑保持；预算 guard 为 epoch1..30、step1..6660。model.install/build 只对 As 行安装此包装器，B00 保留原 NoAs。未上传替换正在运行的 release，也未启动新训练。

验证：9 项本地协议/边界/真实入口声明检查 PASS；N607 真实 practical 引擎的合成 IQ 检查 PASS：E20 旧新输出及完整 metadata 相同，E21/4441 与 E30/6660 有限，epoch/view namespace 刷新，4 个越界负例拒绝，EMA deepcopy 保留自有类型。实际 F 工厂构造的 B00 在 E30/6660 保留 NoAs、0 信道包且 IQ 原样；B01 为 Practical、2 包有限。全部修复验证 0真实IQ/0checkpoint/0NN/0更新/0正式分数。独立定向复核先发现无条件替换会覆盖 NoAs，已改为 c['As'] 条件接入；该问题定向复核 PASS。

原预登记规定技术失败保留产物且不自动重试；用户最新要求不主动扩预算。由于失败 As 无中途权重，补齐需要另从零 E30（额外30模型epoch），不能说成只补10轮。健康 NoAs FINAL30 可作为同预算同seed控制复用，其角色、默认头及冻结依据不变；具体重跑只能在明确处理这项新增计算预算后发布新不可覆盖run，不能覆盖旧失败行。

命名空间兼容补充：自有 channel.py 保留原 tensor_iq 导出（同一个原实现），供 source qualification 和正式预测使用；新增第10项接口检查及真实合成数组转换检查 PASS，0NN。原远端R01未改。
