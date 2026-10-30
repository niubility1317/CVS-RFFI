# 30份源信道抽样的单阶段恢复训练

- run_id：`20261031-phase1-DURA-freshViews-E30-s2026092701-r01`
- group_id：`phase1-dura-scratch-source-draw-count`；类别：`cvs`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：ANALYZED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

紧凑G266951从零单阶段E30/noVar四基础项；原源缓存01..20只读复用，新增原native namespace21..30，后10轮不再复用01..10；无神经教师。

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


## 最小源抽样对照说明

紧凑G266951从零单阶段30/7410/noVar四基础项；唯一变量是30份源信道抽样，原R07的01..20只读链接，21..30沿用同一原native生成器/种子体系/namespace规则生成，不再重复01..10。L6300/U56700/物理ID/顺序/clean单位/源RX-day与固定V面板均保持。30份抽样不是30个新物理总体，clean分支仍可能相同。新10份630000次模拟访问/4.1958GB数值字节，8CPU workers制备，实际成本完整单列；没有神经教师、F/TX/V/target训练读取，不把原R07模型当初始化。

4聚焦检查/唯一P0P1 PASS。controller首步真实2Ls G梯度/严格重载/原F默认头noquerysmoke，随即CPU制备新10份；全部nativeactionreplay/shape/FP32/finite/scene-order/完整计数通过后才GPU6准入训练，期间不占GPU worker。各模块第一步联合更新，B256/.001warm247cos2e-5/WD1e-4/clip1，model/data/probe RNG不变，每轮实际source_cache_view写日志。固定final30后自动七景168000各景/独立truth-last/profile/源V1800，source负结果也不取消。GPU6新增1/max3总PID/free12GB/goal2，其他owner保持。当前待发布，默认与每景3pp/clean/效率目标ACTIVE。


## 实际启动与源数据制备

VERIFIED / RUNNING，当前阶段SOURCE_DATA_PREPARATION，G训练尚未开始。执行32a32998d3c76105b017a0905300e8edbce64e16，4聚焦检查/唯一P0P1 PASS，一次N607 preflight/单archive SHA比较/远端compile通过。首步真实2Ls新G目标梯度、严格保存重载、原F默认头noquerysmoke PASS后立即开始源生成，F/TX/target优化和神经教师读取0。

controller1218619实际CWD/完整argv/commit/CUDA空、8个source spawn子进程及日志增长独立读回；子进程均同一release/CUDA空。当前没有submitted GPU worker，不把CPU制备称为训练。其他owner保持，全部缓存完成后才按GPU6/max3总PID/free12GB/goal2准入。

新增已完成4/10份（21..24），每份63000条/419580000数值字节；共252000次源模拟访问。最近一份52.276848秒，clean9450/受损53550，native action replay最大相对误差2.416e-16，shape/FP32/finite/固定scene顺序均已检查。当前日志4483字节，下一份仍在生成；不据单份时间宣称完整制备或训练加速。

完整10份生成成本、固定30轮、七景正式与匹配profile待返回；没有增加学生预训练或另训头。默认DURA、每景3pp/clean/实际效率目标仍ACTIVE。


## 目标访问前补齐严格随机数配对对照

源首轮差异触发的纯CPU检查证明：旧physical预算匹配，但增强头初始化额外改变CUDA探针seed，先前仅检查CPU RNG的判断不完整。旧数据/分数/权重保留，历史vsphysical只作同预算观察、不当严格单机制归因。补充独立纯G repeatViews E30，与本版stock训练路径/G初值/CUDAprobe2026092701/npshuffle2026103004完全匹配；本版健康32a进程不热改，不改G/loss/预算。初state643张量一致、CUDA未初始化、0G-Fforward/0checkpoint/0TX-V-target的trace见control_cuda_seed_audit.json。新主对照在本版目标访问前登记，独立全七景测试。


## 实际GPU训练核实

VERIFIED / RUNNING，学生训练阶段，执行32a32998d3c76105b017a0905300e8edbce64e16。真实2Ls G目标梯度/严格重载/原F noquery首步PASS。worker1224405/GPU6的实际CWD、完整argv/commit、CUDA、GPU PID及日志增长独立核实。准入已有1PID/free19245MiB，后态2PID。与另一张卡合计2本任务GPU worker/每卡1，其他owner保持。

新增10份源抽样全部完成：630000次模拟访问/4195800000数值字节，CPU8制备530.740140秒；每份原nativeactionreplay/shape/FP32/finite/固定scene顺序检查通过，外部完整元数据读回一致。旧01..20只读链接，源physicalIDs/roles/Vpanel保持；source generation不是训练epoch。当前已读到source_cache_view27，确实没有把21..30映射回01..10。

已27/30轮/6669更新/L170100/U1530900，当前五项加权loss独立复算一致，F/TX/神经teacher/初始化checkpoint训练读取0。当前只证明运行和输入日程，不是正式身份收益；固定final30后自动七景168000/truth-last与profile，默认保持/目标ACTIVE。


## 本行完整七景结果

VERIFIED / ANALYZED，本行30/7410/30不同source cache views、全七景168000/独立126组复算通过。clean84.461905%较raw少360，lowurban66.973810%，七景均低于默认/0of6通过3pp。源制备530.740140s、训练976.251282s、全分析1691.498856s；source额外4.1958GB另列，无神经teacher/F/TX/V/target训练。独立同CUDA探针纯G repeatViews主对照尚在训练，配对增益待完成，不用旧physical混杂对照提前归因。完整见[分析](analysis.md)、results.json。默认保持/目标ACTIVE。
