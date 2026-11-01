# 冻结残差扩散：源端真实采样轨迹保真诊断

- run_id：`20261102-diagnostic-Restore-diffusion-source-path-s2026092701-r01`
- group_id：`restoration-frozen-source-diffusion-sampling-path`；类别：`diagnostic`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：RUNNING（实际源smoke及worker已核实）

## 目的与对照

固定当前final30的received与8步内部clean估计，只用原源512×20面板，无更新或目标IQ/结果读取，不选新部署步数。

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

## 冻结源采样轨迹

本项只分析已经固定的时频残差扩散 final30，不训练新G，也不定义待挑选的部署步数。原Main仍使用固定8步Heun/15denoiser完整采样。诊断只读原R07固定128Ls+384Us共512条、20份既有received及clean参考，TX/V/OE/novel/目标IQ、预测和评分读取均为0；原F0连续特征/defaulthead只作诊断参考，0更新、拟合头或选择。

记录received基准与实际sampler八步内部clean估计：步1至7为各实际state对应denoised估计，步8保留部署末步Euler的真实FP32算术，源2首先检查它逐值等于真正G.forward，并核实实际F0/defaulthead有限，PASS立即全量。诊断不把oracle/noised-clean样本作为模型输入。固定latent来自原checkpoint，推理无新RNG，G/F0最终所有tensor必须不变。

共10240相关sourcecases、92160输出行/F0特征、153600Gcore、512cleanF0参考，源2额外60Gcore/2F0特征。按同条case计算IQ NMSE、F0cos/源方差MSE、Fclean判决一致、输出功率/单位误差和相对首步波形变化，逐step/scene/lock及相对received/首步双向纠错全部报告。Fclean一致不是TX准确率，重复512不视作10240独立包，源内改善不能代替原Main六景3pp与两个clean总体验收，也不证明未知设备保真或恢复极限。

为保护父测试与每卡一个本任务NN，CPU控制器只在父最后GPU阶段cross_both已真实运行在GPU4（/proc PID/cmdline核实），或该阶段exit0后，才在GPU6按free≥12000MiB/原compute≤2/新增1总≤3准入。父未来只剩CPU评分，不会再选择GPU6；诊断与父可并行，整体至多2个本任务NN、各占一张卡。父和其他owner均不修改。只读技术状态，不读父质量；诊断不读取父预测与评分。技术失败保存产物，不自动重试。3项聚焦检查与一次独立P0/P1通过，精确代码/命令/数据句柄/预算/产物/停止规则见登记。


## 实际冻结源诊断已运行

VERIFIED / RUNNING。源2实际G/F0 smoke通过，记录末步逐值等于部署G，已开始逐源draw推理。CPU控制器与worker的PID/CWD/commit/CUDA/GPU和日志增长独立核实；GPU6准入free24110MiB/原0compute/新增1，父最后GPU阶段同时在4。archive单次SHA一致和远端编译通过。0更新/TX/V/OE/目标IQ-预测-评分，完整92160源行待完成复算；不改父部署或选step。
