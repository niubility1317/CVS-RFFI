# DURA 与单阶段对照的源输出整体相位误差分解

- run_id：`20261030-diagnostic-DURA-source-phase-oracle-s2026092701-m2-r01`
- group_id：`phase1-dura-source-output-phase-decomposition`；类别：`diagnostic`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：ANALYZED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

固定source512×clean/缓存01/20×2冻结G；仅相位oracle不拟合幅度/形状；同F特征/原头保持、CPU2/0更新/TX/V/target。

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


## 最小研究说明

仅source固定512（L128/U384），比较原noVar cf40和默认R69，两模型都不更新。每模型先真实2Ls同F checkpoint smoke，再完成clean和原R07缓存01/20三个视图。oracle使用原clean的复相关，只给输出乘单位复数；绝不拟合幅度或改变形状。零相关相位不唯一，记录valid=False，identity仅中性代表，不冒充已估得相位。

相位可消除误差是原IQMSE的精确分项，不是新的参考：设c=mean(r·conj(x))，则原误差与相位对齐误差之差为2(|c|−Re(c))/Ex。该oracle依赖clean，不可部署，也不是TX准确率上界。F指标是同F自身clean选择的一致，不读逐样本TX/V/target。原F知识和G祖先身份保留，当前神经更新0。

仅noVar可支持下一研究：两个缓存视图分别相位有效≥480/512、可消除误差占总IQ误差≥10%、教师选择净增≥6/512、七个源场景均不下降；另clean控制不下降。任一失败不立项该phaseweight训练，不放宽门槛。通过仅允许另登记单阶段scratchE20，将IQMSE中相位分项权重从1变2，原clean和其余损失保持；不部署oracle，不忽略相位，不复活统一相位旋转或公共µ。默认DURA和完整逐景3pp/clean/效率目标不变。


本地9项相关检查通过，唯一独立P0/P1审查PASS；N607预检核实身份/路径。使用既有CPU2环境与原VALIDATED_ONCE源句柄，源码/config以本次Git commit固定，真实F smoke在launcher首步直接继续。


## 完整源端结果

VERIFIED / ANALYZED_SOURCE_ONLY。执行commit `7e2d22ed1fe0feeac1df8381be89e4bb527108c5`，9项相关检查与唯一P0/P1审查PASS；一次preflight、单归档两端SHA和远端compile通过。启动后独立读取实际PID866891/CWD/完整命令/空CUDA，后正常退出；8743字节日志包含先完成两模型真实2Ls smoke，再完整96批/六模型视图。两G/F状态逐项不变、梯度None，神经更新/逐样本TX/V/target读取0。3072个G和6656个F分析输入来自同512物理记录的重复视图，不当作更多独立样本。

|冻结G|源视图|原IQNMSE|相位oracle IQNMSE|可消除误差%|原F一致数|oracle一致数|平均整体相位差°|
|---|---|---:|---:|---:|---:|---:|---:|
|noVar|clean|0.000101540|0.000100587|0.934853|512|512|0.033848|
|noVar|1|0.022352356|0.020081041|10.161418|478|478|1.791391|
|noVar|20|0.021122014|0.019197978|9.109140|496|498|1.771689|
|DURA_default|clean|0.000032950|0.000032793|0.447992|512|512|0.014099|
|DURA_default|1|0.020212826|0.018261772|9.652549|486|487|1.681722|
|DURA_default|20|0.019439708|0.017635867|9.279142|498|499|1.803234|

noVar两个受损视图的整体相位理想修正只移除10.161418%/9.109140%的总IQ误差，F自身clean选择净增0/2条；low suburban或mid urban分别少1条。虽然clean控制保持512/512，但两个视图均未同时达到原条件，phase_weight_study_eligible=False。不启动仅加重整体相位分项的E20训练，不放宽条件或取消场景保护。

默认DURA控制的相位可消除比例为9.652549%/9.279142%，教师选择各只净增1条。当前源输出的平均整体相位差约1.7°至1.8°；这不能由F对45°至315°旋转敏感直接推导出G的整体相位就是主要瓶颈。整体相位以外的剩余误差仍可能含幅度、频率相关相位、时序、滤波与噪声，尚未分离，不能都称为TX细节损伤。

仅相位oracle使用sourceclean，输出幅度与形状不变，全部6个模型视图均满足误差分解与能量检查。它不可部署，不是TX准确率上界；教师一致数也不是实际TX准确率。近乎零的clean误差分项对FP32减法舍入敏感，只用于控制，不按其小百分比立项。独立NumPy从完整pairedIQ与Ffeatures/logits核对72条完整指标向量（36864值），再重算132组/5412项声明的FP32统计，最大向量差2.68784984e-06，资格一致。

计算循环16.605771秒，CPU2/0GPU；模型加载与发布另计，不冒充GPU推理时延。原source512/单seed、原clean参考和F50defaulthead保持，默认DURA和完整逐景3pp/clean/训练效率目标继续。该结果关闭本轮整体相位优先路线；下一研究应依据整体相位以外的残差，不能据本结果改目标分组权重、重选父权重或宣称已经提高正式性能。


## 事后源频谱残差描述

复用同一六个已保存source模型视图的IQ和F选择标记，新增G/F前向、样本、训练及V/target读取均0。每包按其clean FFT能量排序，分四个各64点的等大小分位，Q1最强、Q4最弱；它不是固定频段，也没有按TX标签划分。Parseval和分区求和逐样本核对通过。代码见experiments/pamf_dura_efficiency/posthoc_spectrum.py，完整聚合见posthoc_spectrum.json。

|G|源视图|Q1 clean能量%|Q4 clean能量%|Q4相对NMSE|F原选择保持组NMSE|F原选择变化组NMSE|
|---|---|---:|---:|---:|---:|---:|
|noVar|1|88.477817|0.109379|0.462093|0.015879|0.113357|
|noVar|20|88.477817|0.109379|0.439664|0.018597|0.099410|
|DURA_default|1|88.477817|0.109379|0.402022|0.017088|0.078630|
|DURA_default|20|88.477817|0.109379|0.388324|0.015052|0.175533|

Q4只有约0.109%的clean能量，noVar的Q4相对NMSE约0.44至0.46；全局NMSE会弱化这些误差的占比。但F选择变化组的整体NMSE也明显较高，不能把身份问题全部归因弱频点。F选择标记比较F自身clean输出，不是真实TX正确/错误标记；小组只有14至34条，不作因果或稳定性结论。删除整体相位后，Q4相对误差仍相近，支持转向整体相位以外的残差，而非继续相位优先。

方法历史核对：R02粗恢复实际目标为IQ NMSE+.1对数功率FFT差，没有解析RF支路损失。更早R01的七架构曾加入frequency/PA/默认Sinc/highfreq/reference损失，但主身份资格全部未过；不能称这些约束已经验证有效或被默认DURA遗漏。若在当前单阶段物理展开结构试验，应是独立匹配消融，先用合法源输出检查梯度和成本、固定权重，再按新版本全七景验证；不得只凭弱频点相对误差或历史R01推定成功。
