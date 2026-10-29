# Phase2 Group1：源域拼接增强＋中期普通 PL 发布

Run：`20261030-phase2-group1-concat-midPL-threepool-m4-r01`。用户已授权新发布，旧三个停止 run 保留停止。状态：PREREGISTERED，尚未声明启动成功。

## 科学矩阵

唯一方法/预算配置为 `configs/pamf_phase2_group1_release_20261030.json`，由实际执行的Group1 catalog/recipe生成；不复用旧组合队列。14家族中11已绑定：PAMF、7 DA及Orthogonal/ISSL/LoRa三注册方法。CSIL作者Fisher、Tweak M10、MoPC-HR原编码器输入绑定暂不执行，保留明确缺项。MoPC已有增量核心，但旧通用六层IQ骨干不能冒充原编码器；公开作者网络入口要求3×32×32/1024表示，与当前单份256-IQ缺少已核实转换，不能猜造。

A：各原完整监督目标＋clean/星地拼接，卫星项0.68；中期普通PL0.95/权重1；无新分类头、无EMA；保留原完整固定预算及明确source-only LR规则。纯度量监督/伪标签损失保持原形式。U真值隐藏。A源增强三景40/35/25与30%/60%/80%日程继承项目默认practical实现，源fc2.462GHz；目标仍为r10的六分量混合/2.45GHz，二者不混同。

B只比较旧类适应；C从A直接开始，旧目标标签实际使用数逐方法记录。PAMF需要旧＋新support，外部原生C不加B10。DURA固定v1，support/query四格两次support拟合。AC/ABC相同终态条件按规范特征批次核对；首固定工作点另有3次×2support路径的BC增量资源测量，只有资源输出，不用于候选或query选择。

主表MIX8/三池/7RX/4seed/K5/N20；曲线MIX01/K1,5,10,20×N2,5,10,20；三单景LS/MU/LU各5episode。全部query用补集，最小176。完整14家族拟合上限44184，目前正确绑定部分35616，另6个固定BC资源拟合。44个新source模型、21263个预测任务及21263个独立CPU评分任务；每方法第一seed/P01/MIX01/RX1-1先只闭合K5/N20主点，之后补该cap其他曲线点，主点不重复。后续目标任务依赖该预测闭合，绝不依赖目标分数；source训练可独立并行。该计数不等于optimizer steps或GPU小时。

## 复用与权限

复用已VALIDATED_ONCE的r10三池、source L6300/U56700/V27000角色及冻结DURA。旧CE-only、REPL、PL不同日程不充作本组普通PL中期＋拼接源模型；本次匹配源checkpoint复用数0，避免错误继承。源训练配置只含source句柄，A自身初始/最终真实checkpoint的无query smoke先于目标访问。Source缓存只按原物理ID对齐，不新增数据哈希/许可链。

## 资源及运行条件

N607，8×RTX3090；每GPU至多2个本run训练/预测进程，其他owner另计，启动前实际free≥6000MiB。CPU scorer最多1；独立source增强生产器4个CPU子worker。CWD为不可变release根，环境CVS-RFFI。精确运行句柄与命令在私有runtime/delivery附件；入口为 `python -u -m experiments.pamf_phase2_practical.group1_dispatch --runtime <release_root>/runtime.json`。

每次A有端到端及训练资源文件，B/C有实际fit资源文件，query推理单独计时；记录CPU RSS采样峰、CUDAallocated/reserved、实际优化步数/参数实例、U曝光/接受及类别直方图、source与target实际标签量、G/F/核成本。每5秒保存live资源快照，避免中途停止导致全部记录丢失。并发运行结果标为有争用测量，不假称独占时延或可归因能耗。BC源特征缓存增量与完整冷support计算分别记录，不重复计费。

## 代码、验证、停止与预期产物

代码身份为发布archive对应的Git commit，由runtime记录。提交后仅传一份不可变archive，单次local/remote SHA对比及remote compile；无额外签名/审批/数据重验。发布前执行聚焦协议负测和实际源checkpoint无query smoke；一次独立P0/P1检查只涉及下一真实run会错/越权/破坏/不能运行或不能合法预测的缺陷。

停止规则：源/profile技术失败阻止其后继；至少两行同指纹的系统性pre-prediction错误、源cache服务失败、协议/输出冲突停止新接纳并保留partial。低分不停止、不调参、不重采信道，不修改其他owner。预测单元全部固定后才独立truth-last，分数不控制后续训练/排序。

预期产物：`owner.json`、`method_matrix.json`、`jobs.json`、`pipeline_state.json`、`source/*/group1_complete.json`、`source/*/A_resources.json`、`resource_telemetry/*/A_end_to_end.json`、逐job日志、逐B/C资源及预测、独立scores与逐RX/TX/day指标。当前登记不证明这些产物已产生；启动后按PID/CWD/cmdline/GPU/log增长独立核实。
