# R37：复用 R32 训练产物修复评分，并正式测试两版

R32 两版已完整训练 E20/4940，原 L6300/U56700、累计 L126000/U1134000、三倍波形前向；0 TX 标签、0 F 权重/输出、0 target 进入优化。各版七视图标准源预测及十二个方向响应单元全部保存，均在标签访问前固定。随后 CPU scorer 在 `evaluation.score` 把 `source_reference` 的实际三个返回值按四个解包而抛错，未写源评分或进入 target。两版 worker 和 controller 均已自然退出。该问题已本地同入口重现；修复仅改为 `ids,_,_`，保留真实预测验证先于真值读取。

原 R32 失败目录、权重、预测、日志保持。R36 尚未发布，因要求原 parent ANALYZED 而由本 R37 评测恢复 run 替代。R37 严格检查原 R32 commit、具体异常、全部所属进程退出、两版 E20/4940 完整合法训练和已完成源预测，排除既有 target 输出；在全新不可覆盖目录复制两版最终 checkpoint、源标准预测与响应预测，复用只读原源对照。0 新训练、0 新源模型推理、0 重选 checkpoint，不在原失败 root 补写成功标志。

发布首步以实际两条 L_s IQ 对两版完整 checkpoint 和原 F50 做无 query smoke，通过后立即继续。两版所有七标准源视图和十二响应单元全部验证，再由独立 CPU subprocess 按原算法评分，允许源 V 标签仅用于评分分组，不更新参数或状态。恢复的 response 预测和 summary 保留原 R32 观察 run 标识，R37 另存恢复来源记录，不伪造原预测来源。源评分数值和源资格都不决定是否测试；两版随后均进入原 F50 seed2026092701/default head 的完整正式矩阵。

正式输入仍为原 R05 固定 168000 物理记录 × clean、practical_high、practical_mid、practical_low_suburban、practical_high_urban、practical_mid_urban、practical_low_urban。恢复器只接收当前 IQ、输出一份 IQ，由原 PAMF 一致派生 FFT、PA、时域和统计输入。每版全部七视图 IQ/features/logits/variance/ID 完成后，独立 CPU scorer 才读取 target truth。测试结果不回流训练、选模、门槛或选择性重跑。

报告每版源域及正式结果、三种郊区、三种 urban、clean 损伤、accuracy/Macro-F1、逐 TX/RX/day、repair/damage 和波形误差。low urban ≥70%；其他五个受损场景各自 ≥79.67619047619047%，即距原 raw clean 84.67619047619047% 不超过 5 个百分点。源域改善、局部合成响应或单个 F 的输出均不能证明每个恢复细节属于输入 TX。本次为已暴露固定历史测试，非新的独立盲测。

N607 普通账户，固定 Git release，最多两条本任务 GPU 预测 worker、每卡一条；池 0/1/5/7、既有共享 receiver_residual_capacity4.lock、计入其他 owner 的每卡少于两进程和至少 12 GB 空闲。0 停止或修改 peers。输入、来源、冲突、非有限数或评分/预测技术错误时保留产物并停止安排待启动行，不自动重试；同一错误经本次修复仍复现则停止盲目重启。一次预检、一次 archive 两端 SHA 比较、远端 compile、一次 scoped P0/P1、实际 smoke 和启动后独立 PID/CWD/argv/GPU/log/产物核实按项目最小流程执行。
