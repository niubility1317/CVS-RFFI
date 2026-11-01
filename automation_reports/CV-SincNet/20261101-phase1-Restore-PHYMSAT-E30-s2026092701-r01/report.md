# 恢复性能主线：物理信道估计与双先验联合恢复，从零单阶段30轮

- run_id：`20261101-phase1-Restore-PHYMSAT-E30-s2026092701-r01`
- group_id：`restoration-joint-physical-MSAT-scratch30`；类别：`cvs`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

新8613449参数PHY-MSAT：仅IQ输入估计source-compatible真实线性信道与噪声，两个不共享context-MSAT先验分别8CG，全部从零同次30轮联合训练。原LsUs模拟trace仅源监督，原F0特征指导；固定final30自动完整原Main七景。

## 数据、seed与模型来源

实际数据契约、权限例外、完整seed角色、checkpoint来源和选择规则见experiment.json。
逐行配置通过config_ref/resolved_config_ref定位；待补项必须在对应生命周期补齐。

## 执行与存储

命令、环境、CWD、commit、launch owner、输出和日志路径见experiment.json。
实际PID/GPU、读取时间、remote readback、失败或替代关系在此追加，并用record命令记录证据指针。

## 结果与覆盖

尚无结果。按预登记artifact逐项记录路径和缺项；保留每row与RX/day/TX/scene/K/seed的对应关系。
源域训练完成、预测完成、评分完成及协议有效性分别陈述。不得用总索引或旧状态证明当前运行。

## 交接

记录已完成、当前run/commit、证据路径、阻塞与下一步；恢复先查原run，不重复启动。

## 新系统与本次固定检验

PHY-MSAT 共8613449参数，输入仍只有当前received IQ。新编码器估计64点复数FIR、256点主/镜像系数和正实噪声方差；两次不共享的带context MSAT先验更新分别接8次CG，合计16CG/81次complex FFT-IFFT。不是在旧G后加滤波，全部参数从零同一AdamW、同一30轮日程联合训练，无旧G/coarse/IQ神经教师或分阶段预训练。

仅原63000 L_s/U_s及原R07二十份模拟（后10轮重复前10份）。真实模拟kernel/coefficients/noise只用于源监督：估计H作用于clean与真实H(clean)的前向NMSE权重0.1，噪声log误差权重0.001；小于1e-7的源噪声标签（包括clean哨兵1e-8）在log监督中按1e-7处理，缓存不变。部署不接收clean、真实H或噪声。IQ NMSE、0.05 STFT、原VarF固定源F0特征项及0.5 IQ梯度限幅保持；F0/default head不更新，不读TX/V/OE/目标来训练或选模。

这是结构、物理求解和额外合法模拟监督一起变化的系统候选，不能把差异只归于容量或某个loss。source-only均值检查不支持删RX维；已有真实信道后校正源收益小且混合，也不能预先证明联合物理训练有效。本版固定单模型seed2026092701，从零单阶段30/7410，sourceorder2026110102和原draw/LR/B256，final30唯一。

主验收保持原地面完整Tri-SIG-TIME SSL/default head、原Main split跨RX126000+跨RX日期42000，全部clean+六景预测后独立truth-last。raw clean86.538690%，每景至少83.538690%且clean保持；分总体clean损伤另列，不能用均值遮盖。每个实际训练版本全部测试，不因源分数差取消。默认DURA和旧VarF只作冻结对照，不自动替换。

13项聚焦检查及一次P0/P1审查PASS。首步真实source2 F0+新G/梯度/无更新/E0重载检查，PASS立即继续。每轮原子保存模型/优化器/RNG仅技术恢复，不参与选模；预留5GiB。只选已授权GPU4/6，free>=16000MiB且原进程上限保持；不停止、重启其他owner或管理设备。完整成本实测，不从结构或利用率推定提速。
