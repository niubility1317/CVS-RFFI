# 源端完整 IQ 协方差与真实信道后验诊断

- run_id：`20261102-diagnostic-Restore-source-covariance-inverse-s2026092701-r01`
- group_id：`restoration-source-full-covariance-gaussian-inverse`；类别：`diagnostic`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：ANALYZED（完整源病例及独立复算已核实）

## 目的与对照

原LsUs拟合统一512维实IQ均值/协方差，只在原固定512x20源面板比较全协方差与同trace各向同性Gaussian真Action后验。无G训练或目标访问。

## 数据、seed与模型来源

实际数据契约、权限例外、完整seed角色、checkpoint来源和选择规则见experiment.json。
逐行配置通过config_ref/resolved_config_ref定位；待补项必须在对应生命周期补齐。

## 执行与存储

命令、环境、CWD、commit、launch owner、输出和日志路径见experiment.json。
实际PID/GPU、读取时间、remote readback、失败或替代关系在此追加，并用record命令记录证据指针。

## 结果与覆盖

10240源病例×两个后验路径完整，24组及全部配对复算PASS，预固定源假设条件满足；Fclean一致不是TX准确率，详见analysis.md。
源域训练完成、预测完成、评分完成及协议有效性分别陈述。不得用总索引或旧状态证明当前运行。

## 交接

记录已完成、当前run/commit、证据路径、阻塞与下一步；恢复先查原run，不重复启动。

## 源端相关性先验假设

原合法63000条LsUs clean拟合统一512维实IQ均值与总体协方差（无TX分组），固定C=(1−0.001)C_emp+0.001tauI，tau=trace(C_emp)/512。比较同均值、同trace的tauI与完整C，两条路径都使用原源模拟真实FIR、主/镜像调制与实部噪声方差。以FP64 Cholesky求解mu+C A^T(A C A^T+sigma²I)^−1(y−A mu)，不增加CG预算、数值fallback或按结果调正则。记录源协方差谱，不按谱选择维数。

原固定128Ls+384Us×20份R07为10240相关实例。两个路径全部报告逐scene/lock的IQ NMSE、F0cos、Fclean一致率及对各向同性/DURA/received的同条净变化；DURA和received只复用旧完整源诊断的标量，没有旧G训练权重或额外G推理。F0严格原3ff8地面完整Tri-SIG-TIME SSL/defaulthead冻结，首步真实source2检查dense因果reflect A与已验证Action一致、后验残差与实际F0有限，PASS继续所有路径。最终F状态不变。20480新F0输出、512 clean参考、2条额外smoke；0TX/V/OE/novel/目标访问/optimizer。

这是有额外真实信道信息、在拟合先验的源样本上完成的机制诊断。它不是可部署模型、TX准确率、未见RX结果或恢复上界，不能据此声称达到Main六景3pp。预先固定下一方向的支持标准：完整C低档城区Fclean一致不低于既有DURA且比各向同性至少高3个百分点，同时整体一致不低于DURA。即使满足也只支持另立部署网络实验；不满足则不自动启动该方向，不按结果修改本次shrinkage或补挑样本。

CPU控制器CUDA空，完整SigmaMix结束后固定GPU6准入（free≥12000MiB、existing compute≤2、新增1总≤3）；不动其他owner。一次独立P0/P1、3项针对性检查、源2真实smoke；技术失败保留且不自动重试。完整矩阵/命令/CWD/数据句柄/代码/GPU/停止规则/产物见登记。


## 真实源诊断执行已核实

VERIFIED。实际source2 dense reflect Action与FP64后验/F0 smoke通过，全部source draw诊断已开始。控制器/worker的PID/CWD/commit/CUDA/GPU与日志增长独立核实；GPU6准入free22342MiB/原2compute/新增1，SigmaMix完整结束，无其他本任务NN。实际执行commit `a3a90b55a73702954c453eec9c1ad17fec24d87f`，archive一次SHA一致、远端编译通过。仅拟合原63000源IQ统一均值/C，无G训练或TX/V/OE/novel/目标读取；最终结果待完整病例和独立复算。


## 完整源协方差诊断完成

VERIFIED / ANALYZED。完整C整体Fclean一致96.601563%、低档城区87.743733%，相对DURA同条事件净+149/10240、+55/1436；相对各向同性净+871/10240、+341/1436。源内支持条件全部满足，但真实A/噪声是oracle，不提供Main或恢复上界。全部10240病例/24组/全部配对复算差0，FP64均值/C/谱及tau核实；同各向同性较旧CG128的Fclean事件差0。实际source2/PID-CWD-CUDA-GPU6/loggrowth已捕获，20draw/exit0/liveempty，0G前向/TX/V/OE/novel/目标/optimizer。
