# 连续源特征条件恢复器：从零单阶段30轮

- run_id：`20261101-phase1-Restore-FeatureCond-E30-s2026092701-r01`
- group_id：`restoration-feature-conditioned-single30`；类别：`cvs`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

共享MSAT encoder回归源F0连续160维特征，预测latent经四FiLM调decoder，输出单位复功率。源teacher只loss/部署只IQ；新结构+辅助回归+功率约束系统比较，不是单因素。

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

## 新结构与权限

新G4670178参数全部从零单阶段30轮/7410联合训练。共享MSAT encoder从received预测连续160维原F0源clean特征，四FiLM用预测latent调节bottleneck/decoder，最终IQ投影到原clean单位复功率。teacher特征仅在detach后的loss中出现，前向/部署不读取clean/F0/标签。新latent MLP是连续特征回归，不是6类身份头；原Tri-SIG-TIME SSL/3ff8/E50-11100/defaulthead冻结，不改头/不读TX、logits、CE、类原型，不加载旧G/coarse/DURA或神经IQ教师。

原Ls6300Us56700、R07原20draw加前10份复用/同sourceorder/seed2026092701-data2026110102/B256tail24/AdamW.001-warm247-cos2e-5/WD1e-4/globalclip1保留。loss为IQ NMSE+.05STFT+.1(F0输出cos+源varMSE)+.1(预测latentcos+源varMSE)，无逐记录cap。输出特征与latent目标共用原F0 clean63000缓存及f57源尺度，不在V/目标拟合。无额外phase/gain/FIR训练增强，V/OE/novel/目标不训练、选模或早停，OE仅冻结诊断。结构、辅助回归与输出投影一起变化，非单因素归因。

尊重原clean单位功率，不强制相位/增益等变。源端压力诊断发现+90°会明显改变F0(clean)自身判决，故不以等变误差当Main问题的因果解释；单位功率是源数据约定，不是目标统计。初值为unit_power(received)，不是精确identity，source2实际smoke核投影初值/真实梯度/E0严格重载，PASS立即30训练。所有学习参数随机重建，零旧G祖先；逐轮恢复仅技术用途，final30固定。

主验收仍是原地面完整Tri-SIG-TIME SSL/defaulthead/Main168000物理包的clean+六景，各practical至少140345正确=83.538690%，距raw clean86.538690%≤3pp并保持clean。每次训练冻结后自动126000/42000两总体七景，所有42单元预测固定后独立CPU truth-last/原共享锁，原raw/DURA同IQ-ID预测精确复用。源质量不取消测试，不依据目标反馈改参数；默认DURA保持。

7项针对性检查及一次独立P0/P1通过。静态合成CPU输入profile为260546560实数Conv/Linear/attention MAC（含QKV/out/QK/AV），0FFT/CG；未计F0、FiLM逐点、norm、activation、softmax、插值和内存，不是N607时延。环境原N607 CVS-RFFI Torch2.1+cu121/FP32-TF32off，独占CWD/commit/命令/路径/产物见登记。GPU4/6 UUID free>=12GB、原compute<=2、只新增1个本任务GPUworker/单卡总<=3/自身每卡1/整体<=2，其他owner不改；出错保留root与恢复文件，不自动重试。完整质量与成本仍待真实运行。
