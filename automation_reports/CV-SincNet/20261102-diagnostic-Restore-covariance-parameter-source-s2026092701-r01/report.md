# 冻结相关性恢复器：信道与噪声估计误差的源端归因

- run_id：`20261102-diagnostic-Restore-covariance-parameter-source-s2026092701-r01`
- group_id：`restoration-covariance-frozen-source-parameter-attribution`；类别：`diagnostic`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：ANALYZED（四路源病例及独立复算完整）

## 目的与对照

固定源512x20/final30，在同mu/C下比较全盲、仅真实噪声、仅真实Action及两者真实；无新训练或Main读取，不改变父训练/部署/测试。

## 数据、seed与模型来源

实际数据契约、权限例外、完整seed角色、checkpoint来源和选择规则见experiment.json。
逐行配置通过config_ref/resolved_config_ref定位；待补项必须在对应生命周期补齐。

## 执行与存储

命令、环境、CWD、commit、launch owner、输出和日志路径见experiment.json。
实际PID/GPU、读取时间、remote readback、失败或替代关系在此追加，并用record命令记录证据指针。

## 结果与覆盖

固定10240实例×4路径、48组与所有配对完整；source2盲路径等于实际部署，详见analysis.md。
源域训练完成、预测完成、评分完成及协议有效性分别陈述。不得用总索引或旧状态证明当前运行。

## 交接

记录已完成、当前run/commit、证据路径、阻塞与下一步；恢复先查原run，不重复启动。

## 固定源端参数归因

只分析固定CovarianceMAP final30/7410（bf29554a、1230341参数）与原3ff8地面完整Tri-SIG-TIME SSL/defaulthead。原128Ls+384Us共512源包×20既有R07 draw=10240相关实例，比较四路：估计Action+估计噪声、仅换真实源噪声、仅换真实源Action、二者都真实。四路使用同条received、同G保存的FP32全512维mu/C和同FP64 Cholesky；真实参数只是源模拟oracle，不成为部署变体。

所有路径逐scene/lock报告IQ NMSE、Fcos、Fclean一致率、相对盲/DURA/received的同条修复/损伤/净变化，以及实际矩阵误差和噪声误差。DURA/received只复用旧源诊断标量；G和F所有state最终不变。没有TX/V/OE/novel/目标IQ-预测-评分、优化更新、新统计拟合、换头或选模。原源IQ统计63000和F0 E50祖先知识成本保留；重复512不当10240独立包，Fclean一致不是TX准确率、Main泛化、未知设备证明或严格恢复上限。

首步实际source2盲后验逐值等于G.forward、原F0输出有限，PASS立即四路全量。10240 G估计器记录、40960后验/F0输出、512 clean参考；额外smoke为4估计器/4后验/2 F0记录。3针对性检查与一次独立P0/P1通过。

控制器先CUDA空等待：只在父最后GPU阶段cross_both已经从/proc PID/cmdline核实在4或6运行后，选另一卡；或该阶段exit0后从4/6准入。父后续只有CPU评分，不会回到GPU。实际free≥12000MiB/existing compute≤2/新增1总≤3，整体至多两条本任务NN且分卡。父训练/部署/Main矩阵和其他owner均保持，诊断不按源质量取消或改变父测试。技术失败保留且不自动重试。完整路径/代码/命令/CWD/GPU/停止规则/产物见登记。


## 实际等待控制器已核实

VERIFIED / QUEUED。实际CPU控制器PID/CWD/cmdline/CUDA空与等待日志已核实，尚无submitted子worker或源smoke/结果；不以发布返回值冒充诊断已运行。代码`5d7d9a2c4dd8e5d58b2843e47b1769d93d434c29`，一次archive SHA比对和远端compile通过。父仍按既定30轮和完整Main运行；到最后GPU阶段才在另一卡按预算准入，未知源结果不影响父。


## 四路源端参数归因完整完成

VERIFIED / ANALYZED。盲/真实噪声/真实Action/两者真实，整体Fclean一致依次95.029297/95.058594/96.523438/96.601562%，低档城区依次82.242340/82.660167/87.325905/87.743733%。全10240实例/48组/所有配对独立复算PASS，最大差0，旧source-DURA/received标量与Fclean参考差0；source2盲==G.forward及F0有限、全G/F0状态不变、完整20draw/exit0/liveempty核实。运行中postlaunch worker快照捕获True；源oracle不是部署版本或Main成绩，父矩阵保持。
