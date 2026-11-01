# 冻结扩散第4去噪读出：原Main七景独立推理对照

- run_id：`20261102-evaluation-Restore-DiffusionEarly4-main-s2026092701-r01`
- group_id：`restoration-diffusion-source-motivated-fixes`；类别：`cvs`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

根据合法源轨迹固定第4denoised读出，7core；原G权重不变/0新训练，自动完整Main2pop7/四路预测后truth-last。

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

## 本项仅改变冻结推理读出

严格加载原5e5921c9从零单阶段30/7410的固定final模型，不更新或另训G。全局固定第4denoised估计，前三次完整Heun更新各2denoiser加第4次denoiser，共7调用；不执行第4次correction及后续步。权重、配置与保存latent全部继承，外包模块不增加参数。源2用真实parent和F0检查，且新输出逐值等于源轨迹[3]。

原full8、raw、DURA的同IQ/ID预测精确复用；early4全量新推理，四路各七景/两个总体共56单元完整后评分。新训练成本0，祖先G30轮/F0 E50/源尺度成本仍计入系统，不能称系统没有训练，也不冒充一个新从零网络。1614608参数、7×176971776=1238802432静态实数MAC、21个complex256 FFT/IFFT；不是实测时延或已证明性能收益。
