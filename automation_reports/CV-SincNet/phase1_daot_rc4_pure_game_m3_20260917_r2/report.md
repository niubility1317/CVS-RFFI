# 原生DAOT＋RC4与仅动力博弈三seed对照

本轮21行：原生DAOT＋RC4三seed；SIM/EG/CF-EG/TR-EG/XT-DANN/DRIC仅博弈各三seed。seed392005/392006/392007，全部从零训练E200。逐行配置、seed角色和输出见experiment.json。

仅博弈路径完全跳过DAOT、RC4校准/路由/H/P及自监督和z_dom辅助损失，保留共同基础目标及L/U域对抗头监督，U到编码器对抗梯度仍为0。原旧R0关闭行仍有RC4辅助项，不能替代本轮纯博弈。新18行与已完成联合响应行按方法/seed配对，包含移除整套DAOT/RC4的效应。

原生三行采用F0_FIXED_BASE训练配方，保留AMP和fasttrust调度，不附加响应求解器；源域连续物理ID逐角色核对同一契约。原生训练seed控制多个训练随机流，与新响应入口不同，故为配方比较，不冒充严格单因素消融。

source为RX1/3/4/6/8、day1/2/3，L/U/V=6300/56700/27000。训练不构造目标loader，所有checkpoint来源为空、EMA由本轮学生产生。最终E200冻结后按相同opaque测试包、batch256、eval seed392005做Clean及三LEO评分，单列day0跨天＋跨接收机结果；不据目标成绩改参或重跑。

每GPU最多2个训练，不干预已有健康任务；技术失败暂停后续排队，已有健康子任务继续。尚未完成训练，不能提供新测试结论。

## r2技术修复

r1首行在训练前记录resolved_config时SatViewStage对象不能直接JSON序列化而退出。源域物理角色比对已通过，但未开始训练；dispatcher已退出，其余20行未启动。r1全部保留。r2仅将dataclass日程结构转换为JSON对象，保持所有科学参数、seed与预算不变。真实三seed配置回归：旧写法均复现同异常，新写法均通过无损JSON roundtrip。

## N607启动状态：VERIFIED

{'RUNNING': 13, 'QUEUED': 8}；dispatcher PID=2153663。实际逐行PID/GPU/argv、resolved config与日志见launch_readback_verified.json。启动代码commit=4a36c49914ee8ff922d79b7d54a3098ce8f4daa4，release=/home/szu2070436088/2510044040/CV-SincNet/releases/adv3b02_controls_4a36c49914。原生三seed来源均为scratch；pure_game行明确关闭DAOT/RC4，已观察到接受步。

独立原问题复审PASS：序列化转换不修改原始args或训练配方；三seed回归均通过。r1失败保留，本轮未重用任何失败权重。
