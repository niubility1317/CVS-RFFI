# 部署噪声尺度均衡训练：同初值从零单阶段30轮

- run_id：`20261102-phase1-Restore-DiffusionSigmaMix-E30-s2026092701-r01`
- group_id：`restoration-diffusion-source-motivated-fixes`；类别：`cvs`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：RUNNING（真实源模型检查及实际worker已核实）

## 目的与对照

仅将训练sigma改为50%原lognormal+50%均匀8部署点；同架构/初值/数据/全局Gaussian RNG/损失/30轮/完整15core部署，自动原Main七景。

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

## 源依据、固定比较与权限

原固定源512×20轨迹已完成92160行/108组/全部配对独立复算，误差0；实际末步等于G.forward，G/F0状态不变。整体第4内部读出Fclean一致90.869141%、IQ NMSE .053697，末步87.294922%/.094406；低档城区69.428969%/.142072→62.465181%/.253139。它揭示源内后半段采样损伤，并提出训练noise尺度覆盖问题。Fclean一致不是TX准确率；源探索后固定后续设定，不声称事前盲选，也不以已曝光Main错例或分组调参。

本组两项独立检验分别固定推理读出与训练sigma分布，不因一项结果好坏决定是否跑另一项，不按目标场景拼接模型。原完整八步/15调用模型、DURA及历史结果保持。正式基座仍为原地面完整Tri-SIG-TIME SSL/defaulthead/Main126000+42000各七景，六景每景距同一raw clean≤3个百分点，并保持两个clean总体；性能主线、效率支线。

只有原Ls6300/Us56700、R07 source配对及冻结原F0/3ff8连续160特征/源方差知识。无TX/logits/CE/原型/新身份头、V/OE/novel/目标训练、选模或早停，OE仅冻结诊断；F-guided非F-free，原F0/defaulthead不更新。部署逐记录IQ，固定保存latent，无clean/F0/真实信道/SNR/ID选择/batch统计作为G输入。源2真实模型检查为首个神经工作，PASS即完整矩阵；全部prediction固定后原共享锁独立CPU truth-last，保留全部TX/RX/day/损伤与纠错。单seed、基准曝光及未知设备保真限制均保留，默认DURA不变。

5项聚焦检查及每候选唯一一次独立P0/P1通过。两个run分别固定GPU4和6，每卡一个本任务NN/全局至多2，free≥12000MiB/原compute≤2/新增1总≤3；其他owner不动。一个不可覆盖root/代码commit/归档SHA比较/远端编译/实际PID-CWD-CUDA-GPU-日志核实；技术失败保存产物，无自动重试或质量提前停止。精确命令、数据句柄、预算和产物见登记。

## 本项只改变源训练sigma抽样

全G同原factory随机初值，从零单阶段30/7410；原R07源顺序20draw+前10、model2026092701/order2026110102、B256/247、AdamW.001-warm247-cos2e-5/WD1e-4/clip1、EDM MSE+.05STFT+.1F0cos/源方差MSE和完整8步Heun/15调用部署保持。训练每条pair1denoiser。

先按原全局CUDA seed2026110203抽原lognormal sigma和Gaussian epsilon；再用私有Generator(seed2026110205+当前0基update)，以50%保留原sigma、50%均匀选择固定8个部署sigma。私有抽样不推进原全局随机流，update已在技术快照中保存，恢复时无需额外随机状态。9项mixture计数与5项sigma区间统计、全30日志/顺序/LR/加权loss/30恢复快照完整保留；实际E0与原模型及E01全局CPU/CUDA RNG将独立读回核实。

源2 smoke实际运行CUDA mixture并验证全局CUDA RNG未变，同时检查真实G/F0梯度、完整sampler有限及严格E0重载。零head初值是Gaussian residual prior，不是恒等。固定final30后自动全Main七景，无源质量筛选。1614608参数、训练1core/部署15core、2654576640静态实数MAC及45个complex256 FFT/IFFT；实际训练墙钟/显存另报，静态计数不等于实测时延。


## 源驱动对照实际运行已核实

VERIFIED / RUNNING。源2真实G/F0、CUDA mixture不改变全局RNG与严格E0重载PASS。E01独立CPU读回模型/优化器/RNG，实际E0逐tensor同原full8工厂/真实初值，全部30轮sourceorder/draw相同，E01全局CPU/CUDA RNG状态同原full8，配置仅sigma抽样与版本差异；9项mixture计数/5项sigma区间/加权loss复算PASS。 实际PID/CWD/commit/CUDA/GPU及日志增长核实；GPU6准入free24110MiB/原0compute/新增1，两项分别在4/6同时运行。archive单次SHA和远端编译通过，其他owner保持。原Main完整矩阵自动继续，全prediction固定后独立truth-last，正式性能PENDING，默认DURA与目标ACTIVE保持。
