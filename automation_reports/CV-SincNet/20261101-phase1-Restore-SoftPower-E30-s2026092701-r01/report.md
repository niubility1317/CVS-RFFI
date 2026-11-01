# 放开单位功率投影的源IQ恢复单因素对照

- run_id：`20261101-phase1-Restore-SoftPower-E30-s2026092701-r01`
- group_id：`restoration-learned-power-single30`；类别：`cvs`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：LANDED（CPU等待父Main技术完成；尚未开始G训练）

## 目的与对照

固定PowerAnchor同结构和随机E0，保留.1软RMS约束，删除hard单位功率投影；从零单阶段30并自动原Main七景。先等待Rollout完整Main技术完成，不读其分数。

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

## 源端依据与单因素对照

本项不改变原sourceclean单位复功率约定，只检验G是否应当硬投影输出。给定有噪观测时，单位范数clean的条件均值可以落在单位球内部；纯合成反例为两个正交单位clean等可能，均值功率.5，投影后IQ平方风险反而增大。该数学事实不证明实际源分布中存在同样歧义，不证明F0识别收益或理论恢复百分比。F0可能受益于单位功率支持，最终仍须Main真实TX测试，不能用该反例预判结果。

固定对照为PowerAnchor/42ea/final30，选择依据是它同时具有同网络和.1软RMS项，能隔离hard输出投影；不是按Main排名选择父模型。新G与其4670178参数逐tensor随机初值一致，未加载旧G。只删最后unit_power，输出与aux.raw_restored均为received+残差；同随机权重时，新输出逐值等于原模型投影前raw，新输出再投影则逐值等于原输出。softRMS保留.1×(rawRMS/同条cleanRMS−1)²，vector_norm实现非零处同一数学式、零点有限的次梯度，不加epsilon或额外超参数。

原Ls6300/Us56700、R07完整20draw+前10、model2026092701/order2026110102、B256tail24/247steps、AdamW.001-warm247-cos2e-5/WD1e-4/globalclip1、scratch单阶段30/7410、IQ+.05STFT+.1(outputF0cos+sourcevarMSE)+.1(latentcos+同尺度MSE)+.1软RMS均保持。原F0/3ff8/E50-11100/232993/defaulthead和源尺度f57冻结，只读源continuous160；无TX/logits/CE/原型/新分类头、V/OE/novel/目标更新、早停或选模，OE仅冻结诊断，F-guided非F-free，无旧G或IQ教师祖先。

8针对性检查（新5+PowerAnchor3）及一次独立P0P1 PASS。GPU工作的第一步为真实Ls2的raw恒等初值、原F0/G梯度与严格E0重载，PASS立即全30；E01..30仅技术恢复，final30固定，不以源质量取消测试。推理每记录1core、无额外功率归一、F0或clean作为G输入；原Main126000+42000各七景全部预测固定后独立CPU truth-last/原共享锁，raw/DURA同IQ-ID精确复用，新G全量实际调用。逐景3pp与两个总体clean、TX/RX/day双向纠错、相对raw clean损伤完整报告。基准已有历史曝光、单seed及未知设备保真限制披露，默认DURA保持。

CPU控制器先等当前RolloutBridge整个原Main的技术ARTIFACTS_COMPLETE与无非零terminal，只读技术状态、不读取预测/分数，不因其质量取消本项。等待期间不加载G/F0、不打开IQ、不用GPU；日志root为不可覆盖预约，正式run输出root由既有base controller在进入训练后创建。父技术失败写本run所属logs/queue_terminal.json并抛出原异常，无自动重试。这样不与健康Rollout竞争自身GPU阶段。原GPU4/6 UUID/free≥12000MiB/原compute≤2/add1总≤3准入实际NN时重检，其他owner保持。

静态Conv/Linear/attention仍260546560实数MAC、0FFT/CG，与对照相同；仅删除此前静态计数已排除的逐点投影，沿用同函数主体计数，不称新隔离时延测量。source/F0/缓存知识与1890000训练曝光等成本如实记录；固定30轮不是整个系统无历史训练。实际训练墙钟/显存与正式性能分别报告。精确commit、环境/CWD、命令、句柄、输出与停止规则见登记。


## CPU等待队列已独立核实

VERIFIED / LANDED。所属CPU PID/CWD/命令/CUDA空、等待日志增长核实，GPU上无该PID；当前0新增GPU worker/模型加载/IQ读取/训练更新，真实源2与实际E0检查待首个GPU阶段。等待的是Rollout完整Main技术结束，不是其分数或质量；当前健康父任务、其他owner、数据和默认DURA保持。固定新G从零单阶段30后自动原Main七景，性能尚待验证。
