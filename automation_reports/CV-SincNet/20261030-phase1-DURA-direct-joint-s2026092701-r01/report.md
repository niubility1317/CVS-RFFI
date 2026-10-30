# DURA从零单阶段完整联合训练E20

- run_id：`20261030-phase1-DURA-direct-joint-s2026092701-r01`
- group_id：`phase1-dura-single-stage-efficiency`；类别：`cvs`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

RandomcompleteDURAgraph;singleE20/4940jointpairedIQ/prior/spectral/sourceaction/noise;0learnedteacher/ancestorweights;automaticsevenvieworiginalF50truthlast

用户新增训练效率要求后固定此单seed候选：完整266951参数两阶段推理结构，从随机初始化进行**一次E20/4940联合源训练**，祖先checkpoint与学习教师forward均为0。两阶段指推理展开，不分阶段预训练coarse/action/prior，不执行级联剪枝蒸馏。默认DURA v1.0保留，只在冻结评测时作为参考，不用于本候选初始化或训练。

源监督为原Ls6300/Us56700合法pairedclean，以及同份R07模拟器作用与方差。单次目标为最终IQNMSE＋0.1两个内部prior的IQNMSE均值＋0.05原多尺度STFT＋0.05两个新随机白探针作用误差＋0.001 log方差误差，20轮中常量权重。新增的是明确披露的source模拟器监督，0TX标签/F/V/target，不声称无clean监督。AdamW2e-4→2e-5/warm247/WD1e-4/clip1、batch256，L/U累计126000/1134000曝光，所有组件联合优化，固定final，无源/目标选模或早停。

CPU成本诊断B256单步1.535→1.186秒仅支持实现可运行与成本方向，不能替代GPU耗时、收敛和正式质量。本次累计20轮与旧R69已有约120轮＋20续训不具有相同累计预算；必须如实比较成本和最终质量，不能把无祖先成本写成性能等价已经成立。

训练后自动source1800×7、target168000×7和独立truth-last，同一原F50/defaulthead和同一物理记录，记录rawclean差距、DURA差、逐TX/RX/day和净纠错；每个新版本不因source成绩差取消正式测试。GPU实测训练墙钟/峰值显存、完整曲线、同卡G/G+F的B1/B128成本均报告，质量目标仍为六景分别距rawclean≤3pp并保持clean。

源码已通过无继承父/F训练/错物理合同/目标污染checkpoint拒绝检查、source目标不接收梯度检查、编译及一次P0/P1审查。评估构造模式的既有P1已在此候选首次发布前修复，四项回归PASS、一次该问题定点复审PASS。launcher先真实两Ls梯度、随机checkpoint严格存取和实际冻结F50无query smoke，PASS后继续；不加载旧恢复器初始化。

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
